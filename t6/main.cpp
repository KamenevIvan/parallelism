#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

using std::vector;
using std::cout;
using std::endl;

void apply_boundary_conditions(vector<vector<double>>& grid, int nx, int ny) {
    // Corners: TL=10, TR=20, BR=30, BL=20
    grid[0][0] = 10.0;
    grid[0][nx - 1] = 20.0;
    grid[ny - 1][nx - 1] = 30.0;
    grid[ny - 1][0] = 20.0;

    for (int i = 1; i < nx - 1; ++i) {
        grid[0][i] = 10.0 + (20.0 - 10.0) * i / (nx - 1);
        grid[ny - 1][i] = 30.0 + (20.0 - 30.0) * i / (nx - 1);
    }
    for (int j = 1; j < ny - 1; ++j) {
        grid[j][0] = 20.0 + (10.0 - 20.0) * j / (ny - 1);
        grid[j][nx - 1] = 20.0 + (30.0 - 20.0) * j / (ny - 1);
    }
}

double iterate(vector<vector<double>>& grid, vector<vector<double>>& new_grid, int nx, int ny) {
    double max_diff = 0.0;
    for (int i = 1; i < ny - 1; ++i) {
        for (int j = 1; j < nx - 1; ++j) {
            new_grid[i][j] = 0.25 * (grid[i + 1][j] + grid[i - 1][j] +
                                     grid[i][j + 1] + grid[i][j - 1]);
            double diff = std::abs(new_grid[i][j] - grid[i][j]);
            if (diff > max_diff) max_diff = diff;
        }
    }
    return max_diff;
}

int main(int argc, char** argv) {
    int nx, ny, max_iter;
    double eps;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("nx", po::value<int>(&nx)->default_value(128), "grid width")
        ("ny", po::value<int>(&ny)->default_value(128), "grid height")
        ("eps", po::value<double>(&eps)->default_value(1e-6), "tolerance")
        ("max_iter", po::value<int>(&max_iter)->default_value(1000000), "maximum iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    vector<vector<double>> grid(ny, vector<double>(nx, 0.0));
    vector<vector<double>> new_grid = grid;

    apply_boundary_conditions(grid, nx, ny);
    apply_boundary_conditions(new_grid, nx, ny);

    int iter = 0;
    double error = 0.0;

    do {
        error = iterate(grid, new_grid, nx, ny);
        grid.swap(new_grid);
        iter++;
        if (iter % 100 == 0) {
            cout << "Iteration: " << iter << ", Error: " << error << "\r" << std::flush;
        }
    } while (error > eps && iter < max_iter);

    cout << "\nConverged in " << iter << " iterations with error: " << error << endl;

    // Save to binary file
    std::ofstream fout("result.dat", std::ios::binary);
    for (int i = 0; i < ny; ++i)
        fout.write(reinterpret_cast<char*>(grid[i].data()), nx * sizeof(double));
    fout.close();

    return 0;
}
