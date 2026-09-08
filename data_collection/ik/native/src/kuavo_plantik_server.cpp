#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/diagram_builder.h"
#include "plantIK.h"

namespace {

// Kuavo 5W's plant contains 12 leg joints, one waist joint, 14 arm joints,
// and two head joints. plantIK solves the complete 29-position plant; the
// collection schema can take the arm slice [13:27] from the response.
constexpr int kPlantDof = 29;
constexpr int kPoseValues = 7;  // xyz + xyzw
constexpr int kInputValues = kPlantDof + 2 * kPoseValues;

std::vector<double> parse_line(const std::string& line) {
  std::istringstream stream(line);
  std::vector<double> values;
  double value = 0.0;
  while (stream >> value) {
    values.push_back(value);
  }
  return values;
}

Eigen::Quaterniond quaternion_xyzw(const std::vector<double>& values, int offset) {
  const Eigen::Quaterniond q(
      values[offset + 6], values[offset + 3], values[offset + 4], values[offset + 5]);
  const double norm = q.norm();
  if (!(norm > 1e-12) || !std::isfinite(norm)) {
    throw std::runtime_error("invalid quaternion");
  }
  return q.normalized();
}

void print_failure() {
  std::cout << "0\n" << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: kuavo_plantik_server <kuavo5.urdf>\n";
    return 2;
  }

  try {
    drake::systems::DiagramBuilder<double> builder;
    auto* plant = builder.AddSystem<drake::multibody::MultibodyPlant<double>>(0.001);
    drake::multibody::Parser parser(plant);
    parser.AddModels(std::filesystem::path(argv[1]));
    plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("base_link"));
    plant->Finalize();

    if (plant->num_positions() != kPlantDof) {
      throw std::runtime_error("expected the 29-DoF Kuavo 5W Drake URDF");
    }

    auto diagram = builder.Build();
    auto context = diagram->CreateDefaultContext();
    auto& plant_context = diagram->GetMutableSubsystemContext(*plant, context.get());
    (void)plant_context;

    HighlyDynamic::CoMIK solver(
        plant, {"base_link", "zarm_l7_end_effector", "zarm_r7_end_effector"});
    HighlyDynamic::IKParams params;
    params.pos_cost_weight = 0.0;
    params.constraint_mode = 0;

    std::cout << std::setprecision(17);
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) {
        continue;
      }
      try {
        const auto values = parse_line(line);
        if (values.size() != kInputValues) {
          throw std::runtime_error("expected 43 numeric values");
        }
        for (const double value : values) {
          if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite input");
          }
        }

        Eigen::VectorXd q0(kPlantDof);
        for (int i = 0; i < kPlantDof; ++i) {
          q0[i] = values[i];
        }

        const int left_offset = kPlantDof;
        const int right_offset = kPlantDof + kPoseValues;
        Eigen::Vector3d left_position(
            values[left_offset], values[left_offset + 1], values[left_offset + 2]);
        Eigen::Vector3d right_position(
            values[right_offset], values[right_offset + 1], values[right_offset + 2]);
        HighlyDynamic::FramePoseVec poses{
            {Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()},
            {quaternion_xyzw(values, left_offset), left_position},
            {quaternion_xyzw(values, right_offset), right_position},
        };

        Eigen::VectorXd solution;
        if (!solver.solve(poses, q0, solution, params) || solution.size() != kPlantDof) {
          print_failure();
          continue;
        }

        std::cout << "1";
        for (int i = 0; i < kPlantDof; ++i) {
          std::cout << ' ' << solution[i];
        }
        std::cout << '\n' << std::flush;
      } catch (const std::exception& error) {
        std::cerr << "request failed: " << error.what() << '\n';
        print_failure();
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "initialization failed: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
