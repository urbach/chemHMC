#include "global.hpp"
#include "coulomb_pppm.hpp"
#include "math.h" // for definition of M_PI (value of Pi)
#include "Input_reader.hpp"

Coulomb_pppm::Coulomb_pppm(YAML::Node doc, params_class& params) {
    ewald_accuracy = check_and_assign_value<double>(doc["coulomb"], "accuracy");
    ewald_accuracy *= coulombtokcal; // convert accuracy to internal units
    realspace_cutoff = check_and_assign_value<double>(doc["coulomb"], "cutoff");
    realspace_cutoff_squared=realspace_cutoff*realspace_cutoff;
    r_c = realspace_cutoff;
    r_c2 = r_c*r_c;

    // Coefficients for kspace error estimation from 
    // Deserno, M.; Holm, C.; J. Chem. Phys. 109, 7694–7701 (1998)
    error_coeffs[1][0] = 2.0 / 3.0;
    error_coeffs[2][0] = 1.0 / 50.0;
    error_coeffs[2][1] = 5.0 / 294.0;
    error_coeffs[3][0] = 1.0 / 588.0;
    error_coeffs[3][1] = 7.0 / 1440.0;
    error_coeffs[3][2] = 21.0 / 3872.0;
    error_coeffs[4][0] = 1.0 / 4320.0;
    error_coeffs[4][1] = 3.0 / 1936.0;
    error_coeffs[4][2] = 7601.0 / 2271360.0;
    error_coeffs[4][3] = 143.0 / 28800.0;
    error_coeffs[5][0] = 1.0 / 23232.0;
    error_coeffs[5][1] = 7601.0 / 13628160.0;
    error_coeffs[5][2] = 143.0 / 69120.0;
    error_coeffs[5][3] = 517231.0 / 106536960.0;
    error_coeffs[5][4] = 106640677.0 / 11737571328.0;
}

void Coulomb_pppm::init(const particles_instance& particles) {

    double L[3];
    L[0] = particles.L[0];
    L[1] = particles.L[1];
    L[2] = particles.L[2];
    double V = L[0]*L[1]*L[2];
    auto& h_id = particles.h_id;

    kspace_base[0] = 2.0*M_PI/L[0];
    kspace_base[1] = 2.0*M_PI/L[1];
    kspace_base[2] = 2.0*M_PI/L[2];
    
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

    chargesquaredsum *= coulombtokcal;
    // Estimate a good parameters
    // ratio of timing for real/imaginary parts, averaged over 100 runs
    //double timer_ratio = 0.046304/0.018513; 

    estimate_alpha(particles.N,V);
    init_grid(particles, L);
    tune_alpha(particles);
    allocate_views(particles);
    compute_greensfn_fac();
    compute_rho_fac();

    double tmp;

    for(int i = 0; i < grid_size[0]; i++) {
        // do signed modulo via integer division
        tmp = i - grid_size[0]*(2*i/grid_size[0]);
        fkx(i) = kspace_base[0]*tmp;
    }
    for(int i = 0; i < grid_size[1]; i++) {
        // do signed modulo via integer division
        tmp = i - grid_size[1]*(2*i/grid_size[1]);
        fkx(i) = kspace_base[1]*tmp;
    }
    for(int i = 0; i < grid_size[2]; i++) {
        // do signed modulo via integer division
        tmp = i - grid_size[2]*(2*i/grid_size[2]);
        fkx(i) = kspace_base[2]*tmp;
    }

    compute_greensfn_mesh(L);
    for(int l = 0; l < 10; l++) {
        printf("GREENSFN %d:%f\n",l,h_greensfn(l));
    }
}

double Coulomb_pppm::potential(const particles_instance& particles) {
    Kokkos::Timer coulomb_time;

    time_potential += coulomb_time.seconds();
    double V_total = 0.0;
    return V_total;
}

void Coulomb_pppm::force(const particles_instance& particles,type_f& f) {
    Kokkos::Timer coulomb_time;
    
    time_force += coulomb_time.seconds();
}

void Coulomb_pppm::estimate_alpha(int N, double V) {
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

void Coulomb_pppm::init_grid(const particles_instance& particles, double L[3]) {
    
    // Estimate a value for the grid spacing that is
    // as large as possible while still
    // fulfilling the users accuracy requirements

    double initial_grid_spacing = 1.0/ewald_alpha;
    double kspace_error = ewald_accuracy+1.0;

    grid_spacing[0] = initial_grid_spacing;
    grid_size[0] = static_cast<int> (L[0]/grid_spacing[0]) + 1;
    while (kspace_error > ewald_accuracy) {
        kspace_error = kspace_error_1D(grid_spacing[0], L[0], particles.N);
        grid_size[0]++;
        grid_spacing[0] = L[0]/grid_size[0];
    }

    grid_spacing[1] = initial_grid_spacing;
    grid_size[1] = static_cast<int> (L[1]/grid_spacing[1]) + 1;
    kspace_error = ewald_accuracy+1.0;
    while (kspace_error > ewald_accuracy) {
        kspace_error = kspace_error_1D(grid_spacing[1], L[1], particles.N);
        grid_size[1]++;
        grid_spacing[1] = L[1]/grid_size[1];
    }

    grid_spacing[2] = initial_grid_spacing;
    grid_size[2] = static_cast<int> (L[2]/grid_spacing[2]) + 1;
    kspace_error = ewald_accuracy+1.0;
    while (kspace_error > ewald_accuracy) {
        kspace_error = kspace_error_1D(grid_spacing[2], L[2], particles.N);
        grid_size[2]++;
        grid_spacing[2] = L[2]/grid_size[2];
    }

    // Make grid-size even for fast FFTs
    if (grid_size[0] % 2 != 0) grid_size[0]++;
    if (grid_size[1] % 2 != 0) grid_size[1]++;
    if (grid_size[2] % 2 != 0) grid_size[2]++;

    // Final adjustment
    grid_spacing[0] = L[0]/grid_size[0];
    grid_spacing[1] = L[1]/grid_size[1];
    grid_spacing[2] = L[2]/grid_size[2];
    printf("FFT GRID SIZE: %d %d %d\n",grid_size[0],grid_size[1],grid_size[2]);
    printf("FFT GRID SPACING: %f %f %f\n",grid_spacing[0],grid_spacing[1],grid_spacing[2]);
}

double Coulomb_pppm::kspace_error_1D(double grid_spacing, double box_length, int N) {
    double sum = 0.0;
    for (int i = 0; i < stencil_order; i++) {
        sum += error_coeffs[stencil_order][i] * pow(grid_spacing*ewald_alpha,2.0*i);
    }
    
    double kspace_error = chargesquaredsum * pow(grid_spacing*ewald_alpha,(double)stencil_order) *
                    sqrt(ewald_alpha*box_length*sqrt(2.0*M_PI)*sum/N) / (box_length*box_length);
    return kspace_error;
}

double Coulomb_pppm::total_kspace_error(const particles_instance& particles) {
    double x_error = kspace_error_1D(grid_spacing[0], particles.L[0], particles.N);
    double y_error = kspace_error_1D(grid_spacing[1], particles.L[1], particles.N);
    double z_error = kspace_error_1D(grid_spacing[2], particles.L[2], particles.N);
    return sqrt((x_error*x_error + y_error*y_error + z_error*z_error)/3.0);///sqrt(3.0);
}

double Coulomb_pppm::rspace_kspace_diff(const particles_instance& particles) {
    double rspace_error = 2.0*chargesquaredsum*
            exp(-ewald_alpha*ewald_alpha*realspace_cutoff_squared) /
            sqrt(particles.N*realspace_cutoff*particles.L[0]*particles.L[1]*particles.L[2]);
    double kspace_error = total_kspace_error(particles);
    return rspace_error - kspace_error;
}

double Coulomb_pppm::num_derivative_diff(const particles_instance& particles) {
    double step_size = 1e-6;
    double diff1,diff2,deriv,old_alpha;
    diff1 = rspace_kspace_diff(particles);
    old_alpha = ewald_alpha;
    ewald_alpha += step_size;
    diff2 = rspace_kspace_diff(particles);
    ewald_alpha = old_alpha;
    deriv = (diff2-diff1)/step_size;
    return deriv;
}

void Coulomb_pppm::tune_alpha(const particles_instance& particles) {
    double diff;
    for (int i = 0; i < 1e5; i++) {
        diff = rspace_kspace_diff(particles)/num_derivative_diff(particles);
        ewald_alpha -= diff;
        if (fabs(rspace_kspace_diff(particles)) < 1e-5) return;
    }
}

void Coulomb_pppm::allocate_views(const particles_instance& particles) {

    density = Kokkos::View<double***>("rho", grid_size[0], grid_size[1], grid_size[2]);
    h_density = Kokkos::create_mirror_view(density);

    // reciprocal‐space force components after inverse‐FFT
    vdx_brick = Kokkos::View<double***>("Fx", grid_size[0], grid_size[1], grid_size[2]);
    vdy_brick = Kokkos::View<double***>("Fy", grid_size[0], grid_size[1], grid_size[2]);
    vdz_brick = Kokkos::View<double***>("Fz", grid_size[0], grid_size[1], grid_size[2]);
    h_vdx_brick = Kokkos::create_mirror_view(vdx_brick);
    h_vdy_brick = Kokkos::create_mirror_view(vdy_brick);
    h_vdz_brick = Kokkos::create_mirror_view(vdz_brick);

    // FFT buffers
    density_fft = Kokkos::View<Kokkos::complex<double>>("rho_fft", grid_size[0]*grid_size[1]*grid_size[2]);
    h_density_fft = Kokkos::create_mirror_view(density_fft);
    // Green’s function G(k)
    greensfn = Kokkos::View<double*>("G", grid_size[0]*grid_size[1]*grid_size[2]);
    h_greensfn = Kokkos::create_mirror_view(greensfn);
    // FFT scratch space (real-to-complex / complex-to-real)
    work1 = Kokkos::View<Kokkos::complex<double>>("w1", 2*grid_size[0]*grid_size[1]*grid_size[2]);
    work2 = Kokkos::View<Kokkos::complex<double>>("w2", 2*grid_size[0]*grid_size[1]*grid_size[2]);
    h_work1 = Kokkos::create_mirror_view(work1);
    h_work2 = Kokkos::create_mirror_view(work2);

    // k-vector tables
    fkx = Kokkos::View<double*>("kx", grid_size[0]);
    fky = Kokkos::View<double*>("ky", grid_size[1]);
    fkz = Kokkos::View<double*>("kz", grid_size[2]);
    h_fkx = Kokkos::create_mirror_view(fkx);
    h_fky = Kokkos::create_mirror_view(fky);
    h_fkz = Kokkos::create_mirror_view(fkz);

    // spline interpolation tables
    // charge‐assignment weights
    greensfn_fac = Kokkos::View<double*> ("greensfn_fac",stencil_order);
    rho_coeff = Kokkos::View<double**>("rhoc", stencil_order,stencil_order);
    h_greensfn_fac = Kokkos::create_mirror_view(greensfn_fac);
    h_rho_coeff = Kokkos::create_mirror_view(rho_coeff);

    // force‐gathering weights 
    drho_coeff = Kokkos::View<double**>("drhoc",stencil_order,stencil_order);
    h_drho_coeff = Kokkos::create_mirror_view(drho_coeff);
    // 1D temps
    rho1d = Kokkos::View<double**>("rho1d",3,stencil_order);
    drho1d = Kokkos::View<double**>("drho1d",3,stencil_order);
    h_rho1d = Kokkos::create_mirror_view(rho1d);
    h_drho1d = Kokkos::create_mirror_view(drho1d);
}

void Coulomb_pppm::compute_greensfn_fac() {
    Kokkos::deep_copy(h_greensfn_fac, 0.0);
    h_greensfn_fac(0) = 1.0;

    int j;
    for (int i = 1; i < stencil_order; i++) {
        for (j = i; j > 0; j--) {
            h_greensfn_fac(j) = 4.0 * (h_greensfn_fac(j) * (j - i) * (j - i - 0.5) - h_greensfn_fac(j - 1) * (j - i - 1) * (j - i - 1));
        }
        h_greensfn_fac(0) = 4.0 * (h_greensfn_fac(0) * (j - i) * (j - i - 0.5));
    }

    // Normalization
    size_t invfac = 1;
    for (int i = 1; i < 2 * stencil_order; i++) invfac *= i;
    double gamma_inverse = 1.0 / invfac;

    for (int i = 1; i < stencil_order; i++) {
        h_greensfn_fac(i) *= gamma_inverse;
    }
    Kokkos::deep_copy(greensfn_fac, h_greensfn_fac);
}

void Coulomb_pppm::compute_rho_fac() {
    int j,k,l,m;
    //create temporary array
    double a[stencil_order][2*stencil_order+1];
    double s;

    for (k = -stencil_order; k <= stencil_order; k++)
    {
        for (l = 0; l < stencil_order; l++) 
        {
            a[l][k+stencil_order] = 0.0;
        }
    }

    a[0][stencil_order] = 1.0;
    for (j = 1; j < stencil_order; j++) 
    {
        for (k = -j; k <= j; k += 2) 
        {
            s = 0.0;
            for (l = 0; l < j; l++) 
            {
                a[l+1][k+stencil_order] = (a[l][k+stencil_order+1]-a[l][k+stencil_order-1]) / (l+1);
                s += pow(0.5,(double) l+1) *
                (a[l][k+stencil_order-1] + pow(-1.0,(double) l) * a[l][k+stencil_order+1]) / (l+1);
            }
            a[0][k+stencil_order] = s;
        }
    }

    /*for (k = -stencil_order; k <= stencil_order; k++)
    {
        for (l = 0; l < stencil_order; l++) 
        {
            printf("%d %d: %f\n",l,k,a[l][k+stencil_order]);
        }
    }*/
    m = 0;
    for (k = -(stencil_order-1); k < stencil_order; k += 2) {
        for (l = 0; l < stencil_order; l++) {
            rho_coeff(l,m) = a[l][k+stencil_order];
            //printf("RHO: %d %d: %f\n",l,m-2,rho_coeff(l,m));
        }
        for (l = 1; l < stencil_order; l++) {
            drho_coeff(l-1,m) = l*a[l][k+stencil_order];
            //printf("dRHO: %d %d: %f\n",l,m-2,drho_coeff(l-1,m));
        }
        m++;
    }
}

double Coulomb_pppm::greensfn_denominator(double snx, double sny, double snz) {
  double s = snx + sny + snz;
  double sum = 0.0, term = 1.0;
  for (int l = 0; l < stencil_order; ++l) {
    sum += greensfn_fac(l) * term;
    term *= s;
  }
  return sum;
}

void Coulomb_pppm::compute_greensfn_mesh(double L[3]) {

    double EPS_HOC = 1e-6;

    // how many alias‐images are needed in each dim
    int nbx = static_cast<int>(
        (ewald_alpha * L[0] / (M_PI * grid_size[0])) * pow(-log(EPS_HOC), 0.25));
    int nby = static_cast<int>(
        (ewald_alpha * L[1] / (M_PI * grid_size[1])) * pow(-log(EPS_HOC), 0.25));
    int nbz = static_cast<int>(
        (ewald_alpha * L[2]   / (M_PI * grid_size[2])) * pow(-log(EPS_HOC), 0.25));

    // linear index into greensfn
    int idx = 0;

    
    Kokkos::deep_copy(greensfn,h_greensfn);
}


 /*for (int mz = 0; mz < grid_size[2]; ++mz) {
        // signed index in [-grid_size[2]/2 .. grid_size[2]/2)
        int mper = (mz < grid_size[2]/2) ? mz : mz - grid_size[2];
        double snz = std::sin(0.5 * kspace_base[2] * mper * (L[2]/grid_size[2]));
            snz = snz*snz;

        for (int i = 0; i < grid_size[1]; ++i) {
            int lper = (i < grid_size[1]/2) ? i : i - grid_size[1];
            double sny = std::sin(0.5 * kspace_base[1] * lper * (L[1]/grid_size[1]));
            sny = sny*sny;

            for (int kx = 0; kx < grid_size[0]; ++kx) {
                int kper = (kx < grid_size[0]/2) ? kx : kx - grid_size[0];
                double snx = std::sin(0.5 * kspace_base[0] * kper * (L[0]/grid_size[0]));
                    snx = snx*snx;

                // squared magnitude of k vector
                double kx_phys = kspace_base[0] * kper;
                double ky_phys = kspace_base[1] * lper;
                double kz_phys = kspace_base[2] * mper;
                double sqk = kx_phys*kx_phys + ky_phys*ky_phys + kz_phys*kz_phys;

                if (sqk == 0.0) {
                h_greensfn(idx++) = 0.0;
                } else {
                // the bare Coulomb prefactor 4π / k^2
                double numerator = 4.0*M_PI / sqk;

                // denominator from spline‐stencil
                double denom = greensfn_denominator(snx, sny, snz);
                if (denom == 0.0) {
                    printf("Denominator zero at idx=%d: snx=%g sny=%g snz=%g\n",idx,snx,sny,snz);
                }

               // alias‐sum correction
                double sum1 = 0.0;
                for (int ax = -nbx; ax <= nbx; ++ax) {
                double qx = kspace_base[0] * (kper + grid_size[0]*ax);
                double sx = std::exp(-0.25 * (qx/ewald_alpha)*(qx/ewald_alpha));
                double argx = 0.5 * qx * (L[0]/grid_size[0]);
                double wx_win = (argx == 0.0)
                                ? 1.0
                                : std::sin(argx)/argx;
                double wx = std::pow(wx_win, stencil_order);

                for (int ay = -nby; ay <= nby; ++ay) {
                    double qy = kspace_base[1] * (lper + grid_size[1]*ay);
                    double sy = std::exp(-0.25 * (qy/ewald_alpha)*(qy/ewald_alpha));
                    double argy = 0.5 * qy * (L[1]/grid_size[1]);
                    double wy_win = (argy == 0.0)
                                    ? 1.0
                                    : std::sin(argy)/argy;
                    double wy = std::pow(wy_win, stencil_order);

                    for (int az = -nbz; az <= nbz; ++az) {
                    double qz = kspace_base[2] * (mper + grid_size[2]*az);
                    double sz = std::exp(-0.25 * (qz/ewald_alpha)*(qz/ewald_alpha));
                    double argz = 0.5 * qz * (L[2]/grid_size[2]);
                    double wz_win = (argz == 0.0)
                                    ? 1.0
                                    : std::sin(argz)/argz;
                    double wz = std::pow(wz_win, stencil_order);

                    // dot(k, q) / |q|^2 factor
                    double dot1 = kx_phys*qx + ky_phys*qy + kz_phys*qz;
                    double q2   = qx*qx + qy*qy + qz*qz;

                    sum1 += (dot1/q2) * sx*sy*sz * wx*wy*wz;
                    }
                }
                }

                h_greensfn(idx++) = numerator * (sum1/denom);
                }
            }
        }
    }*/