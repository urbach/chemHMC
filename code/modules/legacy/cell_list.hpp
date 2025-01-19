
// cell_list related stuff
    Kokkos::View<int*> cells_per_dim;
    Kokkos::View<int*>::HostMirror h_cells_per_dim;
    Kokkos::View<double*> cell_size;   // Size of each cell
    Kokkos::View<double*>::HostMirror h_cell_size;
    Kokkos::View<int**> cell_list;
    Kokkos::View<int**>::HostMirror h_cell_list;
    Kokkos::View<int*> cell_count; // Store the number of particles in each cell
    Kokkos::View<int*>::HostMirror h_cell_count;

// potential energy calculation
    struct Tag_potential_cell {};
    double potential_cell_list();
    KOKKOS_FUNCTION void operator() (Tag_potential_cell, const member_type& teamMember, double& V) const;

    struct Tag_force_cell {};
    void compute_force_cell_list();

    KOKKOS_FUNCTION void operator() (Tag_force_cell, const member_type& teamMember) const;

    // Cell list
    void init_cell_list(YAML::Node& doc);
    void populate_cell_list();
    int compute_cell_index(double x, double y, double z) const;

    struct Tag_populate_cell_list {};

    KOKKOS_FUNCTION void operator()(Tag_populate_cell_list, const int i) const;
    KOKKOS_INLINE_FUNCTION int compute_neighbor_cell_index(int cell_index, int dx, int dy, int dz) const;