#ifndef MODEL_H
#define MODEL_H

#include <pybind11/eigen.h>  // Import needed to pass Eigen variables to python
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <string>

namespace py = pybind11;

namespace ur5CartPole {
    class FCModel {
        public:
            explicit FCModel();
            ~FCModel();

            Eigen::VectorXd predict(const Eigen::VectorXd &obs);

            std::tuple<Eigen::VectorXd, Eigen::VectorXd> forward_traj(const Eigen::VectorXd &obs, const int &n_steps);
        private:
            py::module_ calc_;
            std::string path_;
            py::object model_;
    };
}
#endif // MODEL_H