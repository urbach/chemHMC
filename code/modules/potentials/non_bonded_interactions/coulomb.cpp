#include "global.hpp"
#include "coulomb.hpp"
#include "math.h" // for definition of M_PI (value of Pi)
#include "Input_reader.hpp"

Coulomb::Coulomb(YAML::Node doc, params_class& params) {
    ewald_accuracy = check_and_assign_value<double>(doc["coulomb"], "accuracy");
    ewald_alpha = check_and_assign_value<double>(doc["coulomb"], "alpha");
    sqrt_ewald_alpha = sqrt(ewald_alpha);
    r_c = check_and_assign_value<double>(doc["coulomb"], "cutoff");
    r_c2 = r_c * r_c;
}

void Coulomb::init(const particles_instance& particles) {

    double L[3];
    L[0] = particles.L[0];
    L[1] = particles.L[1];
    L[2] = particles.L[2];
    
    charge = Kokkos::View<double*>("charge", particles.h_atom_type_list.extent(0));
    h_charge = Kokkos::create_mirror_view(charge);

    //generate mapping from atom id to charge
    for (int i = 0; i < particles.h_atom_type_list.extent(0); i++) {
        h_charge(i) = particles.h_atom_type_list(i).charge;
    }
    Kokkos::deep_copy(charge, h_charge);

    //pre-compute sum of charges for later calculations
    chargesum = 0.0;
    for (int i = 0; i < h_charge.extent(0); i++) {
        chargesum += h_charge(i);
    }
    chargesum_sq = chargesum * chargesum;

    // Estimate a good value for alpha

    // ratio of timing for real/imaginary parts, averaged over 100 runs
    //double timer_ratio = 0.046304/0.018513; 
    //ewald_alpha = pow(timer_ratio*(M_PI*M_PI*M_PI*particles.N)/(pow(particles.L[0],6)),(1.0/3.0));

    // the cube root of the timer ratio is ~ 1.36. This is used to estimate alpha
    // if the system is small (few atoms, small box), this can lead to tiny values of alpha
    // that will make the computation slow. In this case a different formula is used. 
    /*ewald_alpha = ewald_accuracy * sqrt(particles.N*r_c*L[0]*L[1]*L[2]) / (2.0*chargesum_sq);
    if (ewald_alpha >= 1.0) ewald_alpha = (1.36 - 0.15*log(ewald_accuracy))/r_c;
    else ewald_alpha = sqrt(-log(ewald_alpha)) / r_c;
    sqrt_ewald_alpha = sqrt(ewald_alpha);

    kspace_base[0] = 2.0*M_PI/L[0];
    kspace_base[1] = 2.0*M_PI/L[1];
    kspace_base[2] = 2.0*M_PI/L[2];

    double x_error;
    kmax_x = 1;
    x_error = 2.0*chargesum_sq*r_c/L[0]
                *sqrt(1.0/(M_PI*particles.N))
                *exp(-M_PI*M_PI/(ewald_alpha*ewald_alpha*L[0]*L[0]));
    while (x_error > ewald_accuracy) {
        kmax_x++;
        x_error = 2.0*chargesum_sq*r_c/L[0]
        *sqrt(1.0/(M_PI*kmax_x*particles.N))
        *exp(-M_PI*M_PI*kmax_x*kmax_x/(ewald_alpha*ewald_alpha*L[0]*L[0]));
    }

    double y_error;
    kmax_y = 1;
    y_error = 2.0*chargesum_sq*r_c/L[1]
                *sqrt(1.0/(M_PI*particles.N))
                *exp(-M_PI*M_PI/(ewald_alpha*ewald_alpha*L[1]*L[1]));
    while (y_error > ewald_accuracy) {
        kmax_y++;
        y_error = 2.0*chargesum_sq*r_c/L[1]
        *sqrt(1.0/(M_PI*kmax_y*particles.N))
        *exp(-M_PI*M_PI*kmax_y*kmax_y/(ewald_alpha*ewald_alpha*L[1]*L[1]));
    }

    double z_error;
    kmax_z = 1;
    z_error = 2.0*chargesum_sq*r_c/L[2]
                *sqrt(1.0/(M_PI*particles.N))
                *exp(-M_PI*M_PI/(ewald_alpha*ewald_alpha*L[2]*L[2]));
    while (z_error > ewald_accuracy) {
        kmax_z++;
        z_error = 2.0*chargesum_sq*r_c/L[2]
        *sqrt(1.0/(M_PI*kmax_z*particles.N))
        *exp(-M_PI*M_PI*kmax_z*kmax_z/(ewald_alpha*ewald_alpha*L[2]*L[2]));
    }

    if (kmax_x > kmax_y) k_max = kmax_x;
    else k_max = kmax_y;
    if (kmax_z > k_max) k_max = kmax_z;
    k_max3d = 4.0*k_max*k_max*k_max+6.0*k_max*k_max+3.0*k_max;

    // magnitudes of largest recpirocal lattice vectors in each direction
    double x_max_magnitude_sq = kspace_base[0]*kspace_base[0]*kmax_x*kmax_x;
    double y_max_magnitude_sq = kspace_base[1]*kspace_base[1]*kmax_y*kmax_y; 
    double z_max_magnitude_sq = kspace_base[2]*kspace_base[2]*kmax_z*kmax_z; 
    if (x_max_magnitude_sq > y_max_magnitude_sq) k_max_magnitude_sq = x_max_magnitude_sq;
    else k_max_magnitude_sq = y_max_magnitude_sq;
    if (z_max_magnitude_sq > k_max_magnitude_sq) k_max_magnitude_sq = z_max_magnitude_sq;

    // We now have all the data we need to allocate memory for the main data
    pot_coeffs = Kokkos::View<double*>("pot_coeffs", k_max3d);
    h_pot_coeffs = Kokkos::create_mirror_view(pot_coeffs);
    force_coeffs = Kokkos::View<double*[3]>("force_coeffs", k_max3d);
    h_force_coeffs = Kokkos::create_mirror_view(force_coeffs);

    kvec_x = Kokkos::View<int*>("kvec_x", k_max3d);
    h_kvec_x = Kokkos::create_mirror_view(kvec_x);
    kvec_y = Kokkos::View<int*>("kvec_y", k_max3d);
    h_kvec_y = Kokkos::create_mirror_view(kvec_y);
    kvec_z = Kokkos::View<int*>("kvec_z", k_max3d);
    h_kvec_z = Kokkos::create_mirror_view(kvec_z);

    // Allocate views for the structure factor calculations (needed later)
    cos_coeffs = Kokkos::View<double***>("cos_coeffs", (2.0*k_max+1,3,particles.N));
    sin_coeffs = Kokkos::View<double***>("sin_coeffs", (2.0*k_max+1,3,particles.N));
    h_cos_coeffs = Kokkos::create_mirror_view(cos_coeffs);
    h_sin_coeffs = Kokkos::create_mirror_view(sin_coeffs);

    // compute coefficients
    double inverse_alpha_sq = 1.0/(ewald_alpha*ewald_alpha);
    double pre_vol = 4.0*M_PI/(L[0]*L[1]*L[2]);
    
    int kcount=0;
    double k_sq;

    for (int m = 1; m <= k_max; m++) {
        // get squared kspace vector magnitude in x-direction
        k_sq = m*kspace_base[0];
        k_sq *= k_sq;
        if (k_sq <= k_max_magnitude_sq) {
            h_kvec_x(kcount) = m;
            h_kvec_y(kcount) = 0;
            h_kvec_z(kcount) = 0;
            h_pot_coeffs(kcount) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
            h_force_coeffs(kcount,0) = 2.0*kspace_base[0]*m*h_pot_coeffs(kcount);
            h_force_coeffs(kcount,1) = 0.0;
            h_force_coeffs(kcount,2) = 0.0;
            kcount++;
        }
        // get squared kspace vector magnitude in y-direction
        k_sq = m*kspace_base[1];
        k_sq *= k_sq;
        if (k_sq <= k_max_magnitude_sq) {
            h_kvec_x(kcount) = 0;
            h_kvec_y(kcount) = m;
            h_kvec_z(kcount) = 0;
            h_pot_coeffs(kcount) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
            h_force_coeffs(kcount,0) = 0.0;
            h_force_coeffs(kcount,1) = 2.0*kspace_base[1]*m*h_pot_coeffs(kcount);
            h_force_coeffs(kcount,2) = 0.0;
            kcount++;
        }
        // get squared kspace vector magnitude in z-direction
        k_sq = m*kspace_base[2];
        k_sq *= k_sq;
        if (k_sq <= k_max_magnitude_sq) {
            h_kvec_x(kcount) = 0;
            h_kvec_y(kcount) = 0;
            h_kvec_z(kcount) = m;
            h_pot_coeffs(kcount) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
            h_force_coeffs(kcount,0) = 0.0;
            h_force_coeffs(kcount,1) = 0.0;
            h_force_coeffs(kcount,2) = 2.0*kspace_base[2]*m*h_pot_coeffs(kcount);
            kcount++;
        }
    }

    for (int k = 1; k <= kmax_x; k++) {
        for (int l = 1; l <= kmax_y; l++) {
            k_sq = (kspace_base[0]*k) * (kspace_base[0]*k);
            k_sq += (kspace_base[1]*l) * (kspace_base[1]*l);
            if (k_sq <= k_max_magnitude_sq) {
                h_kvec_x(kcount) = k;
                h_kvec_y(kcount) = l;
                h_kvec_z(kcount) = 0;
                h_pot_coeffs(kcount) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(kcount,0) = 2.0*kspace_base[0]*k*h_pot_coeffs(kcount);
                h_force_coeffs(kcount,1) = 2.0*kspace_base[1]*l*h_pot_coeffs(kcount);
                h_force_coeffs(kcount,2) = 0.0;
                kcount++;
        
                h_kvec_x(kcount) = k;
                h_kvec_y(kcount) = -l;
                h_kvec_z(kcount) = 0;
                h_pot_coeffs(kcount) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(kcount,0) = 2.0*kspace_base[0]*k*h_pot_coeffs(kcount);
                h_force_coeffs(kcount,1) = -2.0*kspace_base[1]*l*h_pot_coeffs(kcount);
                h_force_coeffs(kcount,2) = 0.0;
                kcount++;
            }
        }
    }

    for (int l = 1; l <= kmax_y; l++) {
        for (int m = 1; m <= kmax_z; m++) {
            k_sq = (kspace_base[1]*l) * (kspace_base[1]*l);
            k_sq += (kspace_base[2]*m) * (kspace_base[2]*m);
            if (k_sq <= k_max_magnitude_sq) {
                h_kvec_x(kcount) = 0;
                h_kvec_y(kcount) = l;
                h_kvec_z(kcount) = m;
                h_pot_coeffs(kcount) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(kcount,0) =  0.0;
                h_force_coeffs(kcount,1) =  2.0*kspace_base[1]*l*h_pot_coeffs(kcount);
                h_force_coeffs(kcount,2) =  2.0*kspace_base[2]*m*h_pot_coeffs(kcount);
                kcount++;
        
                h_kvec_x(kcount) = 0;
                h_kvec_y(kcount) = l;
                h_kvec_z(kcount) = -m;
                h_pot_coeffs(kcount) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(kcount,0) =  0.0;
                h_force_coeffs(kcount,1) =  2.0*kspace_base[1]*l*h_pot_coeffs(kcount);
                h_force_coeffs(kcount,2) = -2.0*kspace_base[2]*m*h_pot_coeffs(kcount);
                kcount++;
            }
        }
    }
        
    for (int k = 1; k <= kmax_x; k++) {
        for (int m = 1; m <= kmax_z; m++) {
            k_sq = (kspace_base[0]*k) * (kspace_base[0]*k);
            k_sq += (kspace_base[2]*m) * (kspace_base[2]*m);
            if (k_sq <= k_max_magnitude_sq) {
                h_kvec_x(kcount) = k;
                h_kvec_y(kcount) = 0;
                h_kvec_z(kcount) = m;
                h_pot_coeffs(kcount) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(kcount,0) =  2.0*kspace_base[0]*k*h_pot_coeffs(kcount);
                h_force_coeffs(kcount,1) =  0.0;
                h_force_coeffs(kcount,2) =  2.0*kspace_base[2]*m*h_pot_coeffs(kcount);
                kcount++;
        
                h_kvec_x(kcount) = k;
                h_kvec_y(kcount) = 0;
                h_kvec_z(kcount) = -m;
                h_pot_coeffs(kcount) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(kcount,0) =  2.0*kspace_base[0]*k*h_pot_coeffs(kcount);
                h_force_coeffs(kcount,1) =  0.0;
                h_force_coeffs(kcount,2) = -2.0*kspace_base[2]*m*h_pot_coeffs(kcount);
                kcount++;
            }
        }
    }
    
    for (int k = 1; k <= kmax_x; k++) {
        for (int l = 1; l <= kmax_y; l++) {
            for (int m = 1; m <= kmax_z; m++) {
                k_sq = (kspace_base[0]*k) * (kspace_base[0]*k);
                k_sq += (kspace_base[1]*l) * (kspace_base[1]*l);
                k_sq += (kspace_base[2]*m) * (kspace_base[2]*m);
                if (k_sq <= k_max_magnitude_sq) {
                h_kvec_x(kcount) = k;
                h_kvec_y(kcount) = l;
                h_kvec_z(kcount) = m;
                h_pot_coeffs(kcount) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(kcount,0) = 2.0*kspace_base[0]*k*h_pot_coeffs(kcount);
                h_force_coeffs(kcount,1) = 2.0*kspace_base[1]*l*h_pot_coeffs(kcount);
                h_force_coeffs(kcount,2) = 2.0*kspace_base[2]*m*h_pot_coeffs(kcount);
                kcount++;
        
                h_kvec_x(kcount) = k;
                h_kvec_y(kcount) = -l;
                h_kvec_z(kcount) = m;
                h_pot_coeffs(kcount) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(kcount,0) = 2.0*kspace_base[0]*k*h_pot_coeffs(kcount);
                h_force_coeffs(kcount,1) = -2.0*kspace_base[1]*l*h_pot_coeffs(kcount);
                h_force_coeffs(kcount,2) = 2.0*kspace_base[2]*m*h_pot_coeffs(kcount);
                kcount++;
        
                h_kvec_x(kcount) = k;
                h_kvec_y(kcount) = l;
                h_kvec_z(kcount) = -m;
                h_pot_coeffs(kcount) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(kcount,0) = 2.0*kspace_base[0]*k*h_pot_coeffs(kcount);
                h_force_coeffs(kcount,1) = 2.0*kspace_base[1]*l*h_pot_coeffs(kcount);
                h_force_coeffs(kcount,2) = -2.0*kspace_base[2]*m*h_pot_coeffs(kcount);
                kcount++;
        
                h_kvec_x(kcount) = k;
                h_kvec_y(kcount) = -l;
                h_kvec_z(kcount) = -m;
                h_pot_coeffs(kcount) = pre_vol*exp(-0.25*k_sq*inverse_alpha_sq)/k_sq;
                h_force_coeffs(kcount,0) = 2.0*kspace_base[0]*k*h_pot_coeffs(kcount);
                h_force_coeffs(kcount,1) = -2.0*kspace_base[1]*l*h_pot_coeffs(kcount);
                h_force_coeffs(kcount,2) = -2.0*kspace_base[2]*m*h_pot_coeffs(kcount);
                kcount++;
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

    printf("alpha: %f   k_max: %d", ewald_alpha, k_max);*/
}

double Coulomb::potential(const particles_instance& particles) {
    Kokkos::Timer coulomb_time;
    const double conversion_factor = 332.062934*kcaltointernal; //conversion to amu* A^2/fs^2
    double V_real = compute_ewald_real(particles);
    double V_reciprocal = compute_ewald_reciprocal(particles);
    //self interaction only needs to be computed once
    if (V_self == 0.0) {
        V_self = compute_ewald_self(particles);
    }

    time_potential += coulomb_time.seconds();
    return conversion_factor*(V_real + V_reciprocal + V_self);
}

void Coulomb::force(const particles_instance& particles,type_f& f) {
    Kokkos::Timer coulomb_time;
    compute_ewald_real_forces(particles,f);
    compute_ewald_reciprocal_forces(particles,f);
    time_force += coulomb_time.seconds();
}

double Coulomb::compute_ewald_real(const particles_instance& particles) {
    double V = 0.0;

    // Capture all needed members of particles_instance
    auto& x = particles.x;
    auto& N = particles.N;
    auto& L = particles.L;
    auto& id = particles.id;
    auto& verlet_list = particles.neighbor_list->verlet_list;
    auto& neighbour_count = particles.neighbor_list->neighbour_count;
    auto& inverse_halved_L = particles.inverse_halved_L;
    auto& charge = this->charge;
    auto& sqrt_ewald_alpha = this->sqrt_ewald_alpha;
    auto& r_c2 = this->r_c2;

    // Outer parallel_reduce
    Kokkos::parallel_reduce(
        "ewald-real-potential",
        Kokkos::TeamPolicy<Tag_potential_ewald_real>(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_potential_ewald_real, const Kokkos::TeamPolicy<>::member_type& teamMember, double& V) {
            const int i = teamMember.league_rank();
            double tmpV = 0.0;
            double sqrt_alpha = sqrt_ewald_alpha;
            double qi = charge(id(i));

            // Inner reduction over neighbors
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(teamMember, neighbour_count(i)),
                [&](const int j, double& innerV) {
                    const int particle_j = verlet_list(i, j);
                    const int type_j = id(particle_j);

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
                    double r = sqrt(r2);
                    innerV += qi * qj * erfc(sqrt_alpha * r) / r;
                    }
                },
            tmpV);

            Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
                V += tmpV;
            });
        },
    V);

    return 0.5*V;
}

double Coulomb::compute_ewald_reciprocal(const particles_instance& particles) {
    double V = 0.0;

    // Capture all needed members of particles_instance
    auto& x = particles.x;
    auto& N = particles.N;
    auto& L = particles.L;
    auto& id = particles.id;
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
                inner_re += charge(id(j)) * cos(kr);
                inner_im += charge(id(j)) * sin(kr);
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

double Coulomb::compute_ewald_self(const particles_instance& particles) {
    double V = 0.0;

    // Capture all needed members of particles_instance
    auto& N = particles.N;
    auto& id = particles.id;
    auto& charge = this->charge;
    auto& ewald_alpha = this->ewald_alpha;

    // Outer parallel_reduce
    Kokkos::parallel_reduce(
        "ewald_self",
        Kokkos::TeamPolicy<Tag_potential_ewald_self>(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_potential_ewald_self, const Kokkos::TeamPolicy<>::member_type& teamMember, double& V) {
            const int i = teamMember.league_rank();

            double qi = charge(id(i));
            double self_energy_contribution = qi * qi;

            Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
                V += self_energy_contribution;
            });
        },
    V);

    return -sqrt(ewald_alpha/M_PI) * V;
}

void Coulomb::compute_ewald_real_forces(const particles_instance& particles,type_f& f) {
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
    auto& sqrt_ewald_alpha = this->sqrt_ewald_alpha;
    auto& ewald_alpha = this->ewald_alpha;
    auto& r_c2 = this->r_c2;

    typedef Kokkos::TeamPolicy<Tag_force_ewald_real> team_policy;
    Kokkos::parallel_for(
        "compute_force_ewald_real",
        team_policy(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_force_ewald_real, const Kokkos::TeamPolicy<>::member_type& teamMember) {
            const int i = teamMember.league_rank();
            double alpha = ewald_alpha;
            double sqrt_alpha = sqrt_ewald_alpha;
            double qi = charge(id(i));

            // Temporary force accumulators for atom i
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, neighbour_count(i)), [=](const int j, double& fx_tmp, double& fy_tmp, double& fz_tmp) {
                if (i != j) {
                    const int particle_j = verlet_list(i, j);
                    const int type_j = id(particle_j);

                    double qj = charge(type_j);
                    double rij[3];

                    // Calculate distance vector and apply minimum image convention
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
                        double erfc_alpha_r = erfc(sqrt_alpha * r);
                        double exp_alpha2_r2 = exp(-alpha * r2);

                        double force_prefactor = qi * qj * (2.0 * sqrt_alpha / sqrt(M_PI) * exp_alpha2_r2 + erfc_alpha_r / r) / (r2);

                        // Accumulate forces
                        fx_tmp += force_prefactor * rij[0];
                        fy_tmp += force_prefactor * rij[1];
                        fz_tmp += force_prefactor * rij[2];
                    }
                }
            }, fx_i, fy_i, fz_i);

            // Update forces on atom i
            Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
                Kokkos::atomic_add(&f(i, 0), fx_i);
                Kokkos::atomic_add(&f(i, 1), fy_i);
                Kokkos::atomic_add(&f(i, 2), fz_i);
            });
        }
    );
    Kokkos::fence();
}

void Coulomb::compute_ewald_reciprocal_forces(const particles_instance& particles,type_f& f) {
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
                Kokkos::atomic_add(&f(i, 0), fx / (L[0] * L[1] * L[2]));
                Kokkos::atomic_add(&f(i, 1), fy / (L[0] * L[1] * L[2]));
                Kokkos::atomic_add(&f(i, 2), fz / (L[0] * L[1] * L[2]));
            });
        }
    );
    Kokkos::fence();
}