#include "global.hpp"
#include "coulomb_ewald.hpp"
#include "math.h" // for definition of M_PI (value of Pi)
#include "Input_reader.hpp"

Coulomb_ewald::Coulomb_ewald(YAML::Node doc, params_class& params) {
    ewald_accuracy = check_and_assign_value<double>(doc["coulomb"], "accuracy");
    ewald_accuracy *= coulombtointernal; // convert accuracy to internal units
    realspace_cutoff = check_and_assign_value<double>(doc["coulomb"], "cutoff");
    realspace_cutoff_squared=realspace_cutoff*realspace_cutoff;
    r_c = realspace_cutoff;
    r_c2 = r_c*r_c;
}

void Coulomb_ewald::init(const particles_instance& particles) {

    double L[3];
    L[0] = particles.L[0];
    L[1] = particles.L[1];
    L[2] = particles.L[2];
    double V = L[0]*L[1]*L[2];
    auto& h_id = particles.h_id;
    
    charge = Kokkos::View<double*>("charge", particles.h_atom_type_list.extent(0));
    h_charge = Kokkos::create_mirror_view(charge);

    //generate mapping from atom id to charge
    for (int i = 0; i < particles.h_atom_type_list.extent(0); i++) {
        h_charge(i) = particles.h_atom_type_list(i).charge;
    }
    Kokkos::deep_copy(charge, h_charge);

    //pre-compute sum of charges for later calculations
    chargesum = 0.0;
    for (int i = 0; i < particles.N; i++) {
        chargesum += h_charge(h_id(i));
        chargesquaredsum += h_charge(h_id(i))*h_charge(h_id(i));
    }
    chargesquaredsum *= coulombtointernal;
    // Estimate a good parameters
    // ratio of timing for real/imaginary parts, averaged over 100 runs
    //double timer_ratio = 0.046304/0.018513; 

    estimate_alpha(particles.N,V);
    estimate_k_max(L,particles.N);

    // We now have all the data we need to allocate memory for the views
    pot_coeffs = Kokkos::View<double*>("pot_coeffs", k_max_total);
    h_pot_coeffs = Kokkos::create_mirror_view(pot_coeffs);
    force_coeffs = Kokkos::View<double*[3]>("force_coeffs", k_max_total);
    h_force_coeffs = Kokkos::create_mirror_view(force_coeffs);

    kvec_x = Kokkos::View<int*>("kvec_x", k_max_total);
    h_kvec_x = Kokkos::create_mirror_view(kvec_x);
    kvec_y = Kokkos::View<int*>("kvec_y", k_max_total);
    h_kvec_y = Kokkos::create_mirror_view(kvec_y);
    kvec_z = Kokkos::View<int*>("kvec_z", k_max_total);
    h_kvec_z = Kokkos::create_mirror_view(kvec_z);

    real_strucfacs = Kokkos::View<double*>("h_real_strucfacs", k_max_total);
    h_real_strucfacs = Kokkos::create_mirror_view(h_real_strucfacs);

    imag_strucfacs = Kokkos::View<double*>("h_imag_strucfacs", k_max_total);
    h_imag_strucfacs = Kokkos::create_mirror_view(h_imag_strucfacs);

    // Allocate views for the structure factor calculations (needed later)
    cos_coeffs = Kokkos::View<double***>("cos_coeffs", 2.0*k_max+1,3,particles.N);
    sin_coeffs = Kokkos::View<double***>("sin_coeffs", 2.0*k_max+1,3,particles.N);
    h_cos_coeffs = Kokkos::create_mirror_view(cos_coeffs);
    h_sin_coeffs = Kokkos::create_mirror_view(sin_coeffs);

    e_field = Kokkos::View<double*[3]>("e_field",particles.N);
    h_e_field = Kokkos::create_mirror_view(e_field);

    pre_compute_coefficients(V);
}

double Coulomb_ewald::potential(const particles_instance& particles) {
    Kokkos::Timer coulomb_time;
    
    // Compute k-space part of ewald sum
    double V_reciprocal = compute_ewald_reciprocal(particles);    
    
    // Compute real-space part of ewald sum
    double V_real = compute_ewald_real(particles);

    // Remove self-interaction
    V_self = compute_ewald_self();
    // Convert to internal units
    double V_total = (0.5*(V_reciprocal+V_real)-V_self) * coulombtointernal;

    time_potential += coulomb_time.seconds();
    return V_total;
}

void Coulomb_ewald::force(const particles_instance& particles,type_f& f) {
    Kokkos::Timer coulomb_time;
    compute_ewald_real_forces(particles,f);
    compute_ewald_reciprocal_forces(particles,f);
    time_force += coulomb_time.seconds();
}

void Coulomb_ewald::compute_structure_factors(const particles_instance& particles) {

    int i,k,l,m,n,ic;
    double k_magnitude_squared,clpm,slpm;
    int N = particles.N;
    auto& h_id = particles.h_id;

    n = 0;

    single_axis_structure_factors(particles, n);

    double real_sf_k_l,imag_sf_k_l,real_sf_k_minus_l,imag_sf_k_minus_l;

    for (k = 1; k <= kmax_x; k++) {
        for (l = 1; l <= kmax_y; l++) {
        k_magnitude_squared = (k*kspace_base[0] * k*kspace_base[0]) + (l*kspace_base[1] * l*kspace_base[1]);
        if (k_magnitude_squared <= k_max_magnitude_squared) {
            real_sf_k_l = 0.0;
            imag_sf_k_l = 0.0;
            real_sf_k_minus_l = 0.0;
            imag_sf_k_minus_l = 0.0;
            for (i = 0; i < N; i++) {
            real_sf_k_l += h_charge(h_id(i))*(h_cos_coeffs(k_max+k,0,i)*h_cos_coeffs(k_max+l,1,i) - h_sin_coeffs(k_max+k,0,i)*h_sin_coeffs(k_max+l,1,i));
            imag_sf_k_l += h_charge(h_id(i))*(h_sin_coeffs(k_max+k,0,i)*h_cos_coeffs(k_max+l,1,i) + h_cos_coeffs(k_max+k,0,i)*h_sin_coeffs(k_max+l,1,i));
            real_sf_k_minus_l += h_charge(h_id(i))*(h_cos_coeffs(k_max+k,0,i)*h_cos_coeffs(k_max+l,1,i) + h_sin_coeffs(k_max+k,0,i)*h_sin_coeffs(k_max+l,1,i));
            imag_sf_k_minus_l += h_charge(h_id(i))*(h_sin_coeffs(k_max+k,0,i)*h_cos_coeffs(k_max+l,1,i) - h_cos_coeffs(k_max+k,0,i)*h_sin_coeffs(k_max+l,1,i));
            }
            h_real_strucfacs(n) = real_sf_k_l;
            h_imag_strucfacs(n++) = imag_sf_k_l;
            h_real_strucfacs(n) = real_sf_k_minus_l;
            h_imag_strucfacs(n++) = imag_sf_k_minus_l;
        }
        }
    }

    double real_sf_l_m,imag_sf_l_m,real_sf_l_minus_m,imag_sf_l_minus_m;

    for (l = 1; l <= kmax_y; l++) {
        for (m = 1; m <= kmax_z; m++) {
        k_magnitude_squared = (l*kspace_base[1] * l*kspace_base[1]) + (m*kspace_base[2] * m*kspace_base[2]);
        if (k_magnitude_squared <= k_max_magnitude_squared) {
            real_sf_l_m = 0.0;
            imag_sf_l_m = 0.0;
            real_sf_l_minus_m = 0.0;
            imag_sf_l_minus_m = 0.0;
            for (i = 0; i < N; i++) {
            real_sf_l_m += h_charge(h_id(i))*(h_cos_coeffs(k_max+l,1,i)*h_cos_coeffs(k_max+m,2,i) - h_sin_coeffs(k_max+l,1,i)*h_sin_coeffs(k_max+m,2,i));
            imag_sf_l_m += h_charge(h_id(i))*(h_sin_coeffs(k_max+l,1,i)*h_cos_coeffs(k_max+m,2,i) + h_cos_coeffs(k_max+l,1,i)*h_sin_coeffs(k_max+m,2,i));
            real_sf_l_minus_m += h_charge(h_id(i))*(h_cos_coeffs(k_max+l,1,i)*h_cos_coeffs(k_max+m,2,i) + h_sin_coeffs(k_max+l,1,i)*h_sin_coeffs(k_max+m,2,i));
            imag_sf_l_minus_m += h_charge(h_id(i))*(h_sin_coeffs(k_max+l,1,i)*h_cos_coeffs(k_max+m,2,i) - h_cos_coeffs(k_max+l,1,i)*h_sin_coeffs(k_max+m,2,i));
            }
            h_real_strucfacs(n) = real_sf_l_m;
            h_imag_strucfacs(n++) = imag_sf_l_m;
            h_real_strucfacs(n) = real_sf_l_minus_m;
            h_imag_strucfacs(n++) = imag_sf_l_minus_m;
        }
        }
    }

    double real_sf_k_m,imag_sf_k_m,real_sf_k_minus_m,imag_sf_k_minus_m;

    for (k = 1; k <= kmax_x; k++) {
        for (m = 1; m <= kmax_z; m++) {
        k_magnitude_squared = (k*kspace_base[0] * k*kspace_base[0]) + (m*kspace_base[2] * m*kspace_base[2]);
        if (k_magnitude_squared <= k_max_magnitude_squared) {
            real_sf_k_m = 0.0;
            imag_sf_k_m = 0.0;
            real_sf_k_minus_m = 0.0;
            imag_sf_k_minus_m = 0.0;
            for (i = 0; i < N; i++) {
            real_sf_k_m += h_charge(h_id(i))*(h_cos_coeffs(k_max+k,0,i)*h_cos_coeffs(k_max+m,2,i) - h_sin_coeffs(k_max+k,0,i)*h_sin_coeffs(k_max+m,2,i));
            imag_sf_k_m += h_charge(h_id(i))*(h_sin_coeffs(k_max+k,0,i)*h_cos_coeffs(k_max+m,2,i) + h_cos_coeffs(k_max+k,0,i)*h_sin_coeffs(k_max+m,2,i));
            real_sf_k_minus_m += h_charge(h_id(i))*(h_cos_coeffs(k_max+k,0,i)*h_cos_coeffs(k_max+m,2,i) + h_sin_coeffs(k_max+k,0,i)*h_sin_coeffs(k_max+m,2,i));
            imag_sf_k_minus_m += h_charge(h_id(i))*(h_sin_coeffs(k_max+k,0,i)*h_cos_coeffs(k_max+m,2,i) - h_cos_coeffs(k_max+k,0,i)*h_sin_coeffs(k_max+m,2,i));
            }
            h_real_strucfacs(n) = real_sf_k_m;
            h_imag_strucfacs(n++) = imag_sf_k_m;
            h_real_strucfacs(n) = real_sf_k_minus_m;
            h_imag_strucfacs(n++) = imag_sf_k_minus_m;
        }
        }
    }

    double real_sf_k_l_m,imag_sf_k_l_m,real_sf_k_minus_l_m,imag_sf_k_minus_l_m;
    double real_sf_k_l_minus_m,imag_sf_k_l_minus_m,real_sf_k_minus_l_minus_m,imag_sf_k_minus_l_minus_m;

    for (k = 1; k <= kmax_x; k++) {
        for (l = 1; l <= kmax_y; l++) {
        for (m = 1; m <= kmax_z; m++) {
            k_magnitude_squared = (k*kspace_base[0] * k*kspace_base[0]) + (l*kspace_base[1] * l*kspace_base[1]) +
            (m*kspace_base[2] * m*kspace_base[2]);
            if (k_magnitude_squared <= k_max_magnitude_squared) {
            real_sf_k_l_m = 0.0;
            imag_sf_k_l_m = 0.0;
            real_sf_k_minus_l_m = 0.0;
            imag_sf_k_minus_l_m = 0.0;
            real_sf_k_l_minus_m = 0.0;
            imag_sf_k_l_minus_m = 0.0;
            real_sf_k_minus_l_minus_m = 0.0;
            imag_sf_k_minus_l_minus_m = 0.0;
            for (i = 0; i < N; i++) {
                clpm = h_cos_coeffs(k_max+l,1,i)*h_cos_coeffs(k_max+m,2,i) - h_sin_coeffs(k_max+l,1,i)*h_sin_coeffs(k_max+m,2,i);
                slpm = h_sin_coeffs(k_max+l,1,i)*h_cos_coeffs(k_max+m,2,i) + h_cos_coeffs(k_max+l,1,i)*h_sin_coeffs(k_max+m,2,i);
                real_sf_k_l_m += h_charge(h_id(i))*(h_cos_coeffs(k_max+k,0,i)*clpm - h_sin_coeffs(k_max+k,0,i)*slpm);
                imag_sf_k_l_m += h_charge(h_id(i))*(h_sin_coeffs(k_max+k,0,i)*clpm + h_cos_coeffs(k_max+k,0,i)*slpm);

                clpm = h_cos_coeffs(k_max+l,1,i)*h_cos_coeffs(k_max+m,2,i) + h_sin_coeffs(k_max+l,1,i)*h_sin_coeffs(k_max+m,2,i);
                slpm = -h_sin_coeffs(k_max+l,1,i)*h_cos_coeffs(k_max+m,2,i) + h_cos_coeffs(k_max+l,1,i)*h_sin_coeffs(k_max+m,2,i);
                real_sf_k_minus_l_m += h_charge(h_id(i))*(h_cos_coeffs(k_max+k,0,i)*clpm - h_sin_coeffs(k_max+k,0,i)*slpm);
                imag_sf_k_minus_l_m += h_charge(h_id(i))*(h_sin_coeffs(k_max+k,0,i)*clpm + h_cos_coeffs(k_max+k,0,i)*slpm);

                clpm = h_cos_coeffs(k_max+l,1,i)*h_cos_coeffs(k_max+m,2,i) + h_sin_coeffs(k_max+l,1,i)*h_sin_coeffs(k_max+m,2,i);
                slpm = h_sin_coeffs(k_max+l,1,i)*h_cos_coeffs(k_max+m,2,i) - h_cos_coeffs(k_max+l,1,i)*h_sin_coeffs(k_max+m,2,i);
                real_sf_k_l_minus_m += h_charge(h_id(i))*(h_cos_coeffs(k_max+k,0,i)*clpm - h_sin_coeffs(k_max+k,0,i)*slpm);
                imag_sf_k_l_minus_m += h_charge(h_id(i))*(h_sin_coeffs(k_max+k,0,i)*clpm + h_cos_coeffs(k_max+k,0,i)*slpm);

                clpm = h_cos_coeffs(k_max+l,1,i)*h_cos_coeffs(k_max+m,2,i) - h_sin_coeffs(k_max+l,1,i)*h_sin_coeffs(k_max+m,2,i);
                slpm = -h_sin_coeffs(k_max+l,1,i)*h_cos_coeffs(k_max+m,2,i) - h_cos_coeffs(k_max+l,1,i)*h_sin_coeffs(k_max+m,2,i);
                real_sf_k_minus_l_minus_m += h_charge(h_id(i))*(h_cos_coeffs(k_max+k,0,i)*clpm - h_sin_coeffs(k_max+k,0,i)*slpm);
                imag_sf_k_minus_l_minus_m += h_charge(h_id(i))*(h_sin_coeffs(k_max+k,0,i)*clpm + h_cos_coeffs(k_max+k,0,i)*slpm);
            }
            h_real_strucfacs(n) = real_sf_k_l_m;
            h_imag_strucfacs(n++) = imag_sf_k_l_m;
            h_real_strucfacs(n) = real_sf_k_minus_l_m;
            h_imag_strucfacs(n++) = imag_sf_k_minus_l_m;
            h_real_strucfacs(n) = real_sf_k_l_minus_m;
            h_imag_strucfacs(n++) = imag_sf_k_l_minus_m;
            h_real_strucfacs(n) = real_sf_k_minus_l_minus_m;
            h_imag_strucfacs(n++) = imag_sf_k_minus_l_minus_m;
            }
        }
        }
    }
}

void Coulomb_ewald::single_axis_structure_factors(const particles_instance& particles, int& n) {
    int i,k,l,m,ic;
    double real_sf_single,imag_sf_single;
    double k_magnitude_squared,clpm,slpm;
    int N = particles.N;
    auto& h_id = particles.h_id;

    for (ic = 0; ic < 3; ic++) {
        k_magnitude_squared = kspace_base[ic]*kspace_base[ic];
        if (k_magnitude_squared <= k_max_magnitude_squared) {
        real_sf_single = 0.0;
        imag_sf_single = 0.0;
        for (i = 0; i < N; i++) {
            h_cos_coeffs(k_max,ic,i) = 1.0;
            h_sin_coeffs(k_max,ic,i) = 0.0;
            h_cos_coeffs(k_max+1,ic,i) = cos(kspace_base[ic]*particles.h_x(i,ic));
            h_sin_coeffs(k_max+1,ic,i) = sin(kspace_base[ic]*particles.h_x(i,ic));
            h_cos_coeffs(k_max-1,ic,i) = h_cos_coeffs(k_max+1,ic,i);
            h_sin_coeffs(k_max-1,ic,i) = -h_sin_coeffs(k_max+1,ic,i);
            real_sf_single += h_charge(h_id(i))*h_cos_coeffs(k_max+1,ic,i);
            imag_sf_single += h_charge(h_id(i))*h_sin_coeffs(k_max+1,ic,i);
        }
        h_real_strucfacs(n) = real_sf_single;
        h_imag_strucfacs(n++) = imag_sf_single;
        }
    }
    
    for (m = 2; m <= k_max; m++) {
        for (ic = 0; ic < 3; ic++) {
        k_magnitude_squared = m*kspace_base[ic] * m*kspace_base[ic];
        if (k_magnitude_squared <= k_max_magnitude_squared) {
            real_sf_single = 0.0;
            imag_sf_single = 0.0;
            for (i = 0; i < N; i++) {

            h_cos_coeffs(k_max+m,ic,i) = h_cos_coeffs(k_max+m-1,ic,i)*h_cos_coeffs(k_max+1,ic,i) -
                h_sin_coeffs(k_max+m-1,ic,i)*h_sin_coeffs(k_max+1,ic,i);

            h_sin_coeffs(k_max+m,ic,i) = h_sin_coeffs(k_max+m-1,ic,i)*h_cos_coeffs(k_max+1,ic,i) +
                h_cos_coeffs(k_max+m-1,ic,i)*h_sin_coeffs(k_max+1,ic,i);
            h_cos_coeffs(k_max-m,ic,i) = h_cos_coeffs(k_max+m,ic,i);
            h_sin_coeffs(k_max-m,ic,i) = -h_sin_coeffs(k_max+m,ic,i);
            real_sf_single += h_charge(h_id(i))*h_cos_coeffs(k_max+m,ic,i);
            imag_sf_single += h_charge(h_id(i))*h_sin_coeffs(k_max+m,ic,i);
            }
            h_real_strucfacs(n) = real_sf_single;
            h_imag_strucfacs(n++) = imag_sf_single;
        }
        }
    }
}

void Coulomb_ewald::estimate_alpha(int N, double V) {
    // the cube root of the timer ratio is ~ 1.36. This is used to estimate alpha
    // if the system is small (few atoms, small box), this can lead to tiny values of alpha
    // that will make the computation slow. In this case a different formula is used. 
    double cutoff = realspace_cutoff;
    double alpha = ewald_accuracy*sqrt(N*cutoff*V)/(2.0*chargesquaredsum);
    if (alpha >= 1.0) {
        alpha = (1.36 - 0.15*log(ewald_accuracy))/cutoff;
    } 
    else {
        alpha = sqrt(-log(alpha)) / cutoff;   
    }
    ewald_alpha = alpha;
}

void Coulomb_ewald::estimate_k_max(double L[3], int N) {
    
    kspace_base[0] = 2.0*M_PI/L[0];
    kspace_base[1] = 2.0*M_PI/L[1];
    kspace_base[2] = 2.0*M_PI/L[2];

    double x_error;
    kmax_x = 1;
    x_error = 2.0*chargesquaredsum*ewald_alpha/L[0]
                *sqrt(1.0/(M_PI*N))
                *exp(-M_PI*M_PI/(ewald_alpha*ewald_alpha*L[0]*L[0]));
    while (x_error > ewald_accuracy) {
        kmax_x++;
        x_error = 2.0*chargesquaredsum*ewald_alpha/L[0]
        *sqrt(1.0/(M_PI*kmax_x*N))
        *exp(-M_PI*M_PI*kmax_x*kmax_x/(ewald_alpha*ewald_alpha*L[0]*L[0]));
    }

    double y_error;
    kmax_y = 1;
    y_error = 2.0*chargesquaredsum*ewald_alpha/L[1]
                *sqrt(1.0/(M_PI*N))
                *exp(-M_PI*M_PI/(ewald_alpha*ewald_alpha*L[1]*L[1]));
    while (y_error > ewald_accuracy) {
        kmax_y++;
        y_error = 2.0*chargesquaredsum*ewald_alpha/L[1]
        *sqrt(1.0/(M_PI*kmax_y*N))
        *exp(-M_PI*M_PI*kmax_y*kmax_y/(ewald_alpha*ewald_alpha*L[1]*L[1]));
    }

    double z_error;
    kmax_z = 1;
    z_error = 2.0*chargesquaredsum*ewald_alpha/L[2]
                *sqrt(1.0/(M_PI*N))
                *exp(-M_PI*M_PI/(ewald_alpha*ewald_alpha*L[2]*L[2]));
    while (z_error > ewald_accuracy) {
        kmax_z++;
        z_error = 2.0*chargesquaredsum*ewald_alpha/L[2]
        *sqrt(1.0/(M_PI*kmax_z*N))
        *exp(-M_PI*M_PI*kmax_z*kmax_z/(ewald_alpha*ewald_alpha*L[2]*L[2]));
    }

    if (kmax_x > kmax_y) k_max = kmax_x;
    else k_max = kmax_y;
    if (kmax_z > k_max) k_max = kmax_z;
    k_max_total = 4.0*k_max*k_max*k_max+6.0*k_max*k_max+3.0*k_max;

    // magnitudes of largest recpirocal lattice vectors in each direction
    double x_max_magnitude_sq = kspace_base[0]*kspace_base[0]*kmax_x*kmax_x;
    double y_max_magnitude_sq = kspace_base[1]*kspace_base[1]*kmax_y*kmax_y; 
    double z_max_magnitude_sq = kspace_base[2]*kspace_base[2]*kmax_z*kmax_z; 
    if (x_max_magnitude_sq > y_max_magnitude_sq) k_max_magnitude_squared = x_max_magnitude_sq;
    else k_max_magnitude_squared = y_max_magnitude_sq;
    if (z_max_magnitude_sq > k_max_magnitude_squared) k_max_magnitude_squared = z_max_magnitude_sq;

    k_max_magnitude_squared += 1e-6;
}

void Coulomb_ewald::pre_compute_coefficients(double V) {
    double inverse_alpha_sq = 1.0/(ewald_alpha*ewald_alpha);
    double pre_vol = 4.0*M_PI/V;
    
    k_total=0;
    double k_sq;

    for (int k = 1; k <= k_max; k++) {
        // get squared kspace vector magnitude in x-direction
        k_sq = k*kspace_base[0];
        k_sq *= k_sq;
        if (k_sq <= k_max_magnitude_squared) {
            h_kvec_x(k_total) = k;
            h_kvec_y(k_total) = 0;
            h_kvec_z(k_total) = 0;
            h_pot_coeffs(k_total) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
            h_force_coeffs(k_total,0) = 2.0*kspace_base[0]*k*h_pot_coeffs(k_total);
            h_force_coeffs(k_total,1) = 0.0;
            h_force_coeffs(k_total,2) = 0.0;
            k_total++;
        }
        // get squared kspace vector magnitude in y-direction
        k_sq = k*kspace_base[1];
        k_sq *= k_sq;
        if (k_sq <= k_max_magnitude_squared) {
            h_kvec_x(k_total) = 0;
            h_kvec_y(k_total) = k;
            h_kvec_z(k_total) = 0;
            h_pot_coeffs(k_total) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
            h_force_coeffs(k_total,0) = 0.0;
            h_force_coeffs(k_total,1) = 2.0*kspace_base[1]*k*h_pot_coeffs(k_total);
            h_force_coeffs(k_total,2) = 0.0;
            k_total++;
        }
        // get squared kspace vector magnitude in z-direction
        k_sq = k*kspace_base[2];
        k_sq *= k_sq;
        if (k_sq <= k_max_magnitude_squared) {
            h_kvec_x(k_total) = 0;
            h_kvec_y(k_total) = 0;
            h_kvec_z(k_total) = k;
            h_pot_coeffs(k_total) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
            h_force_coeffs(k_total,0) = 0.0;
            h_force_coeffs(k_total,1) = 0.0;
            h_force_coeffs(k_total,2) = 2.0*kspace_base[2]*k*h_pot_coeffs(k_total);
            k_total++;
        }
    }

    for (int k = 1; k <= kmax_x; k++) {
        for (int l = 1; l <= kmax_y; l++) {
            k_sq = (kspace_base[0]*k) * (kspace_base[0]*k);
            k_sq += (kspace_base[1]*l) * (kspace_base[1]*l);
            if (k_sq <= k_max_magnitude_squared) {
                h_kvec_x(k_total) = k;
                h_kvec_y(k_total) = l;
                h_kvec_z(k_total) = 0;
                h_pot_coeffs(k_total) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(k_total,0) = 2.0*kspace_base[0]*k*h_pot_coeffs(k_total);
                h_force_coeffs(k_total,1) = 2.0*kspace_base[1]*l*h_pot_coeffs(k_total);
                h_force_coeffs(k_total,2) = 0.0;
                k_total++;
        
                h_kvec_x(k_total) = k;
                h_kvec_y(k_total) = -l;
                h_kvec_z(k_total) = 0;
                h_pot_coeffs(k_total) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(k_total,0) = 2.0*kspace_base[0]*k*h_pot_coeffs(k_total);
                h_force_coeffs(k_total,1) = -2.0*kspace_base[1]*l*h_pot_coeffs(k_total);
                h_force_coeffs(k_total,2) = 0.0;
                k_total++;
            }
        }
    }

    for (int l = 1; l <= kmax_y; l++) {
        for (int m = 1; m <= kmax_z; m++) {
            k_sq = (kspace_base[1]*l) * (kspace_base[1]*l);
            k_sq += (kspace_base[2]*m) * (kspace_base[2]*m);
            if (k_sq <= k_max_magnitude_squared) {
                h_kvec_x(k_total) = 0;
                h_kvec_y(k_total) = l;
                h_kvec_z(k_total) = m;
                h_pot_coeffs(k_total) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(k_total,0) =  0.0;
                h_force_coeffs(k_total,1) =  2.0*kspace_base[1]*l*h_pot_coeffs(k_total);
                h_force_coeffs(k_total,2) =  2.0*kspace_base[2]*m*h_pot_coeffs(k_total);
                k_total++;
        
                h_kvec_x(k_total) = 0;
                h_kvec_y(k_total) = l;
                h_kvec_z(k_total) = -m;
                h_pot_coeffs(k_total) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(k_total,0) =  0.0;
                h_force_coeffs(k_total,1) =  2.0*kspace_base[1]*l*h_pot_coeffs(k_total);
                h_force_coeffs(k_total,2) = -2.0*kspace_base[2]*m*h_pot_coeffs(k_total);
                k_total++;
            }
        }
    }
        
    for (int k = 1; k <= kmax_x; k++) {
        for (int m = 1; m <= kmax_z; m++) {
            k_sq = (kspace_base[0]*k) * (kspace_base[0]*k);
            k_sq += (kspace_base[2]*m) * (kspace_base[2]*m);
            if (k_sq <= k_max_magnitude_squared) {
                h_kvec_x(k_total) = k;
                h_kvec_y(k_total) = 0;
                h_kvec_z(k_total) = m;
                h_pot_coeffs(k_total) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(k_total,0) =  2.0*kspace_base[0]*k*h_pot_coeffs(k_total);
                h_force_coeffs(k_total,1) =  0.0;
                h_force_coeffs(k_total,2) =  2.0*kspace_base[2]*m*h_pot_coeffs(k_total);
                k_total++;
        
                h_kvec_x(k_total) = k;
                h_kvec_y(k_total) = 0;
                h_kvec_z(k_total) = -m;
                h_pot_coeffs(k_total) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(k_total,0) =  2.0*kspace_base[0]*k*h_pot_coeffs(k_total);
                h_force_coeffs(k_total,1) =  0.0;
                h_force_coeffs(k_total,2) = -2.0*kspace_base[2]*m*h_pot_coeffs(k_total);
                k_total++;
            }
        }
    }
    
    for (int k = 1; k <= kmax_x; k++) {
        for (int l = 1; l <= kmax_y; l++) {
            for (int m = 1; m <= kmax_z; m++) {
                k_sq = (kspace_base[0]*k) * (kspace_base[0]*k);
                k_sq += (kspace_base[1]*l) * (kspace_base[1]*l);
                k_sq += (kspace_base[2]*m) * (kspace_base[2]*m);
                if (k_sq <= k_max_magnitude_squared) {
                h_kvec_x(k_total) = k;
                h_kvec_y(k_total) = l;
                h_kvec_z(k_total) = m;
                h_pot_coeffs(k_total) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(k_total,0) = 2.0*kspace_base[0]*k*h_pot_coeffs(k_total);
                h_force_coeffs(k_total,1) = 2.0*kspace_base[1]*l*h_pot_coeffs(k_total);
                h_force_coeffs(k_total,2) = 2.0*kspace_base[2]*m*h_pot_coeffs(k_total);
                k_total++;
        
                h_kvec_x(k_total) = k;
                h_kvec_y(k_total) = -l;
                h_kvec_z(k_total) = m;
                h_pot_coeffs(k_total) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(k_total,0) = 2.0*kspace_base[0]*k*h_pot_coeffs(k_total);
                h_force_coeffs(k_total,1) = -2.0*kspace_base[1]*l*h_pot_coeffs(k_total);
                h_force_coeffs(k_total,2) = 2.0*kspace_base[2]*m*h_pot_coeffs(k_total);
                k_total++;
        
                h_kvec_x(k_total) = k;
                h_kvec_y(k_total) = l;
                h_kvec_z(k_total) = -m;
                h_pot_coeffs(k_total) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(k_total,0) = 2.0*kspace_base[0]*k*h_pot_coeffs(k_total);
                h_force_coeffs(k_total,1) = 2.0*kspace_base[1]*l*h_pot_coeffs(k_total);
                h_force_coeffs(k_total,2) = -2.0*kspace_base[2]*m*h_pot_coeffs(k_total);
                k_total++;
        
                h_kvec_x(k_total) = k;
                h_kvec_y(k_total) = -l;
                h_kvec_z(k_total) = -m;
                h_pot_coeffs(k_total) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(k_total,0) = 2.0*kspace_base[0]*k*h_pot_coeffs(k_total);
                h_force_coeffs(k_total,1) = -2.0*kspace_base[1]*l*h_pot_coeffs(k_total);
                h_force_coeffs(k_total,2) = -2.0*kspace_base[2]*m*h_pot_coeffs(k_total);
                k_total++;
                }
            }
        }
    }

    // Copy all necessary data to device
    Kokkos::deep_copy(kvec_x,h_kvec_x);
    Kokkos::deep_copy(kvec_y,h_kvec_y);
    Kokkos::deep_copy(kvec_z,h_kvec_z);
    Kokkos::deep_copy(pot_coeffs,h_pot_coeffs);
    Kokkos::deep_copy(force_coeffs,h_force_coeffs);

    /*for (int i = 0; i < kvec_x.extent(0); i++) {
        printf("vec: %d\n", kvec_x(i));
    }*/
}

double Coulomb_ewald::compute_ewald_reciprocal(const particles_instance& particles) {
    double V_reciprocal = 0.0;
    compute_structure_factors(particles);
    for (int k = 0; k < k_total; k++)
        V_reciprocal += h_pot_coeffs(k) * (h_real_strucfacs(k)*h_real_strucfacs(k) +
                    h_imag_strucfacs(k)*h_imag_strucfacs(k));
    return V_reciprocal;
}

double Coulomb_ewald::compute_ewald_real(const particles_instance& particles) {
    double V = 0.0;

    // Capture all needed members of particles_instance
    auto& x = particles.x;
    auto& N = particles.N;
    auto& L = particles.L;
    auto& h_id = particles.h_id;
    auto& verlet_list = particles.neighbor_list->verlet_list;
    auto& neighbour_count = particles.neighbor_list->neighbour_count;
    auto& inverse_halved_L = particles.inverse_halved_L;
    auto& charge = this->charge;
    auto& ewald_alpha = this->ewald_alpha;
    auto& r_c2 = this->r_c2;
    // Outer parallel_reduce
    Kokkos::parallel_reduce(
        "ewald-real-potential",
        Kokkos::TeamPolicy<Tag_potential_ewald_real>(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_potential_ewald_real, const Kokkos::TeamPolicy<>::member_type& teamMember, double& V) {
            const int i = teamMember.league_rank();
            double tmpV = 0.0;
            double alpha = ewald_alpha;
            double qi = charge(h_id(i));

            // Inner reduction over neighbors
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(teamMember, neighbour_count(i)),
                [&](const int j, double& innerV) {
                    const int particle_j = verlet_list(i, j);
                    const int type_j = h_id(particle_j);

                    double qj = charge(type_j);
                    double rij[3];

                    // Calculate distance vector and apply periodic boundary conditions
                    rij[0] = x(i, 0) - x(particle_j, 0);
                    rij[1] = x(i, 1) - x(particle_j, 1);
                    rij[2] = x(i, 2) - x(particle_j, 2);

                    rij[0] -= int(rij[0] * inverse_halved_L[0]) * L[0];
                    rij[1] -= int(rij[1] * inverse_halved_L[1]) * L[1];
                    rij[2] -= int(rij[2] * inverse_halved_L[2]) * L[2];

                    double r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
                    if (r2 <= r_c2) {
                        //printf("%d : \n",i,qi,particle_j,qj);
                        double r = sqrt(r2);
                        double dV = qi * qj * erfc(alpha * r) / r;
                        innerV += dV;
                    }
                },
            tmpV);

            Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
                V += tmpV;
            });
        },
    V);
    return V;
}

double Coulomb_ewald::compute_ewald_self() {
    // Only works for charge-neutral simulation boxes
    double V_self = ewald_alpha*(chargesquaredsum/coulombtointernal)/sqrtPI;
    return V_self;
}

/*double Coulomb_ewald::compute_ewald_reciprocal(const particles_instance& particles) {
    double V = 0.0;

    // Capture all needed members of particles_instance
    auto& x = particles.x;
    auto& N = particles.N;
    auto& L = particles.L;
    auto& h_id = particles.h_id;
    auto& charge = this->charge;
    auto& ewald_alpha = this->ewald_alpha;
    auto& k_max = this->k_max;

    int num_kpoints = (2 * k_max + 1) * (2 * k_max + 1) * (2 * k_max + 1);
    // Outer parallel_reduce
    Kokkos::parallel_reduce(
        "ewald_reciprocal",
        Kokkos::TeamPolicy<Tag_potential_ewald_reciprocal>(num_kpoints, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_potential_ewald_reciprocal, const Kokkos::TeamPolicy<>::member_type& teamMember, double& V) {
            const int league_rank = teamMember.league_rank();
            const int num_kpoints_2D = (2 * k_max + 1) * (2 * k_max + 1);
            double alpha = ewald_alpha;

            const int kz = league_rank / num_kpoints_2D - k_max;
            const int ky = (league_rank % num_kpoints_2D) / (2 * k_max + 1) - k_max;
            const int kx = league_rank % (2 * k_max + 1) - k_max;

            if (kx == 0 && ky == 0 && kz == 0) return; // Skip the k=0 term

            double tmpV = 0.0;
            double kx_real = 2 * M_PI * kx / L[0];
            double ky_real = 2 * M_PI * ky / L[1];
            double kz_real = 2 * M_PI * kz / L[2];
            double k2 = kx_real * kx_real + ky_real * ky_real + kz_real * kz_real;

            double S_re = 0.0, S_im = 0.0;

            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, N), [=](const int j, double& inner_re, double& inner_im) {
                double kr = kx_real * x(j, 0) + ky_real * x(j, 1) + kz_real * x(j, 2);
                inner_re += charge(h_id(j)) * cos(kr);
                inner_im += charge(h_id(j)) * sin(kr);
            }, Kokkos::Sum<double>(S_re), Kokkos::Sum<double>(S_im));

            double exp_factor = exp(-k2 / (4 * alpha));
            tmpV += (S_re * S_re + S_im * S_im) * exp_factor / k2;

            Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
                V += tmpV;
            });
        },
    V);

    return 2 * M_PI * V / (L[0] * L[1] * L[2]);
}

double Coulomb_ewald::compute_ewald_self(const particles_instance& particles) {
    double V = 0.0; // Potential

    // Capture all needed members of particles_instance
    auto& N = particles.N;
    auto& h_id = particles.h_id;
    auto& charge = this->charge;
    auto& ewald_alpha = this->ewald_alpha;

    // Outer parallel_reduce
    Kokkos::parallel_reduce(
        "ewald_self",
        Kokkos::TeamPolicy<Tag_potential_ewald_self>(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_potential_ewald_self, const Kokkos::TeamPolicy<>::member_type& teamMember, double& V) {
            const int i = teamMember.league_rank();

            double qi = charge(h_id(i));
            double self_energy_contribution = qi * qi;

            Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
                V += self_energy_contribution;
            });
        },
    V);

    return -sqrt(ewald_alpha/M_PI) * V;
}*/

void Coulomb_ewald::compute_ewald_reciprocal_forces(const particles_instance& particles,type_f& f) {
    compute_structure_factors(particles);
    
    Kokkos::deep_copy(e_field,0.0);

    int kx,ky,kz;
    double cypz,sypz,exprl,expim,partial;

    for (int k = 0; k < k_total; k++) {
        kx = kmax_x+kvec_x(k);
        ky = kmax_y+kvec_y(k);
        kz = kmax_z+kvec_z(k);

        for (int i = 0; i < particles.N; i++) {
            cypz = h_cos_coeffs(ky,1,i)*h_cos_coeffs(kz,2,i) - h_sin_coeffs(ky,1,i)*h_sin_coeffs(kz,2,i);
            sypz = h_sin_coeffs(ky,1,i)*h_cos_coeffs(kz,2,i) + h_cos_coeffs(ky,1,i)*h_sin_coeffs(kz,2,i);
            exprl = h_cos_coeffs(kx,0,i)*cypz - h_sin_coeffs(kx,0,i)*sypz;
            expim = h_sin_coeffs(kx,0,i)*cypz + h_cos_coeffs(kx,0,i)*sypz;
            partial = expim*h_real_strucfacs(k) - exprl*h_imag_strucfacs(k);
            e_field(i,0) += partial*h_force_coeffs(k,0);
            e_field(i,1) += partial*h_force_coeffs(k,1);
            e_field(i,2) += partial*h_force_coeffs(k,2);
        }
    }

    auto& h_id = particles.h_id;

    for (int i = 0; i < particles.N; i++) {
        f(i,0) += coulombtointernal * charge(h_id(i)) * e_field(i,0);
        f(i,1) += coulombtointernal * charge(h_id(i)) * e_field(i,1);
        f(i,2) += coulombtointernal * charge(h_id(i)) * e_field(i,2);
    }
}

void Coulomb_ewald::compute_ewald_real_forces(const particles_instance& particles,type_f& f) {
    // Capture all needed members of "particles" here. We do not want to reference 
    // any members of "particle" directly inside of the kernel, as the class contains
    // functions that are not device safe. This would trigger a lot of compiler warnings.
    auto& x = particles.x;
    auto& N = particles.N;
    auto& L = particles.L;
    auto& id = particles.id;
    auto& verlet_list = particles.neighbor_list->verlet_list;
    auto& neighbour_count = particles.neighbor_list->neighbour_count;
    auto& inverse_L = particles.inverse_L;
    auto& charge = this->charge;
    auto& ewald_alpha = this->ewald_alpha;
    auto& r_c2 = this->r_c2;

    typedef Kokkos::TeamPolicy<Tag_force_ewald_real> team_policy;
    Kokkos::parallel_for(
        "compute_force_ewald_real",
        team_policy(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_force_ewald_real, const Kokkos::TeamPolicy<>::member_type& teamMember) {
            const int i = teamMember.league_rank();
            double alpha = ewald_alpha;
            double qi = charge(id(i));

            Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, neighbour_count(i)), 
            [=](const int j) {
                const int particle_j = verlet_list(i, j);
                const int type_j = id(particle_j);

                double qj = charge(type_j);
                double rij[3];

                // Calculate distance vector
                rij[0] = x(i, 0) - x(particle_j, 0);
                rij[1] = x(i, 1) - x(particle_j, 1);
                rij[2] = x(i, 2) - x(particle_j, 2);

                // Apply periodic boundary conditions
                rij[0] -= round(rij[0] * inverse_L[0]) * L[0];
                rij[1] -= round(rij[1] * inverse_L[1]) * L[1];
                rij[2] -= round(rij[2] * inverse_L[2]) * L[2];

                double r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];

                if (r2 < r_c2) {
                    double r = sqrt(r2);
                    double erfc_alpha_r = erfc(alpha * r);
                    double exp_alpha2_r2 = exp(-alpha*alpha * r2);
                    double force_prefactor = coulombtointernal*qi * qj * (2.0 * alpha / sqrt(M_PI)
                                                * exp_alpha2_r2 + erfc_alpha_r / r) / (r2);
                    
                    double f_x[3];
                    f_x[0] = force_prefactor * rij[0];
                    f_x[1] = force_prefactor * rij[1];
                    f_x[2] = force_prefactor * rij[2];
                    
                    // Accumulate forces
                    Kokkos::atomic_add(&f(i, 0), f_x[0]);
                    Kokkos::atomic_add(&f(i, 1), f_x[1]);
                    Kokkos::atomic_add(&f(i, 2), f_x[2]);

                    Kokkos::atomic_add(&f(particle_j, 0), -f_x[0]);
                    Kokkos::atomic_add(&f(particle_j, 1), -f_x[1]);
                    Kokkos::atomic_add(&f(particle_j, 2), -f_x[2]);
                }
            });
        }
    );
    Kokkos::fence();
}

/*void Coulomb_ewald::compute_ewald_reciprocal_forces(const particles_instance& particles,type_f& f) {
    // Capture all needed members of "particles" here. We do not want to reference 
    // any members of "particle" directly inside of the kernel, as the class contains
    // functions that are not device safe. This would trigger a lot of compiler warnings.
    auto& x = particles.x;
    auto& N = particles.N;
    auto& L = particles.L;
    auto& id = particles.id;
    auto& charge = this->charge;
    auto& ewald_alpha = this->ewald_alpha;
    auto& k_max = this->k_max;

    int num_kpoints = (2 * k_max + 1) * (2 * k_max + 1) * (2 * k_max + 1);
    typedef Kokkos::TeamPolicy<Tag_force_ewald_reciprocal> team_policy;
    Kokkos::parallel_for(
        "compute_force_ewald_reciprocal",
        team_policy(num_kpoints, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_force_ewald_reciprocal, const Kokkos::TeamPolicy<>::member_type& teamMember) {
            const int league_rank = teamMember.league_rank();
            const int num_kpoints_2D = (2 * k_max + 1) * (2 * k_max + 1);
            double alpha = ewald_alpha;

            const int kz = league_rank / num_kpoints_2D - k_max;
            const int ky = (league_rank % num_kpoints_2D) / (2 * k_max + 1) - k_max;
            const int kx = league_rank % (2 * k_max + 1) - k_max;

            if (kx == 0 && ky == 0 && kz == 0) return; // Skip the k=0 term

            double kx_real = 2 * M_PI * kx / L[0];
            double ky_real = 2 * M_PI * ky / L[1];
            double kz_real = 2 * M_PI * kz / L[2];
            double k2 = kx_real * kx_real + ky_real * ky_real + kz_real * kz_real;

            double exp_factor = exp(-k2 / (4 * alpha)) / k2;

            // Compute the structure factors S_k
            double S_re = 0.0, S_im = 0.0;

            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, N), [=](const int j, double& inner_re, double& inner_im) {
                double kr = kx_real * x(j, 0) + ky_real * x(j, 1) + kz_real * x(j, 2);
                double charge_j = charge(id(j));
                inner_re += charge_j * cos(kr);
                inner_im += charge_j * sin(kr);
            }, Kokkos::Sum<double>(S_re), Kokkos::Sum<double>(S_im));

            // Compute the forces
            Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, N), [=](const int i) {
                double qi = charge(id(i));
                double kr_i = kx_real * x(i, 0) + ky_real * x(i, 1) + kz_real * x(i, 2);

                double sin_kr_i = sin(kr_i);
                double cos_kr_i = cos(kr_i);

                double force_prefactor = 2.0 * qi * exp_factor;

                // Force components
                double fx = force_prefactor * (S_re * sin_kr_i - S_im * cos_kr_i) * kx_real;
                double fy = force_prefactor * (S_re * sin_kr_i - S_im * cos_kr_i) * ky_real;
                double fz = force_prefactor * (S_re * sin_kr_i - S_im * cos_kr_i) * kz_real;

                // Update forces
                Kokkos::atomic_add(&f(i, 0), coulombtointernal*fx / (L[0] * L[1] * L[2]));
                Kokkos::atomic_add(&f(i, 1), coulombtointernal*fy / (L[0] * L[1] * L[2]));
                Kokkos::atomic_add(&f(i, 2), coulombtointernal*fz / (L[0] * L[1] * L[2]));
            });
        }
    );
    Kokkos::fence();
}*/