#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/String.h>
#include <vector>
#include <queue>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <trajectory_msgs/MultiDOFJointTrajectoryPoint.h>

// Global variables
std::vector<int> indices;
bool indices_received = false;
bool new_indices_received = false;
bool trajectory_planned = false;
bool trajectory_published = false;

std::vector<geometry_msgs::PoseStamped> points;
std::vector<geometry_msgs::PoseStamped> trajectory;
std::vector<geometry_msgs::PoseStamped> caminho;
std::vector<geometry_msgs::PoseStamped> final_path;
size_t current_waypoint_index = 0;
geometry_msgs::PoseStamped current_pose;
bool first_point_published = false;
double total_planned_distance = 0.0;

// Global variables for start and end points
geometry_msgs::PoseStamped start_point;
geometry_msgs::PoseStamped end_point;

// Distancia euclidiana
double euclideanDistance(const geometry_msgs::Pose& p1, const geometry_msgs::Pose& p2) {
    return std::sqrt(std::pow(p1.position.x - p2.position.x, 2) +
                           std::pow(p1.position.y - p2.position.y, 2) +
                           std::pow(p1.position.z - p2.position.z, 2));
}

// Verifica se ha colisao
bool isInCollision(const geometry_msgs::Pose& pose) {
    std::vector<std::tuple<geometry_msgs::Point, double, double, double>> boxes;

    geometry_msgs::Point box1, box2, box3;
    box1.x = 4.0; box1.y = 13.5; box1.z = 4.5;
    box2.x = 10.0; box2.y = 13.5; box2.z = 4.5;
    box3.x = 16.0; box3.y = 13.5; box3.z = 4.5;

    boxes.push_back(std::make_tuple(box1, 3.5, 23.0, 9.0));
    boxes.push_back(std::make_tuple(box2, 3.5, 23.0, 9.0));
    boxes.push_back(std::make_tuple(box3, 3.5, 23.0, 9.0));

    const double safety_margin = 0.05;  // Add a safety margin

    for (const auto& box : boxes) {
        geometry_msgs::Point center;
        double width, height, depth;
        std::tie(center, width, height, depth) = box;

        if (pose.position.x >= center.x - (width / 2 + safety_margin) && pose.position.x <= center.x + (width / 2 + safety_margin) &&
            pose.position.y >= center.y - (height / 2 + safety_margin) && pose.position.y <= center.y + (height / 2 + safety_margin) &&
            pose.position.z >= center.z - (depth / 2 + safety_margin) && pose.position.z <= center.z + (depth / 2 + safety_margin)) {
            return true;
        }
    }
    return false;
}

// Se houver colisao adiciona pontos de transicao
std::vector<geometry_msgs::Pose> findNeighbors(const geometry_msgs::Pose& pose) {
    std::vector<geometry_msgs::Pose> neighbors;
    const double step = 0.7;  // Smaller step size
    const std::vector<std::tuple<double, double, double>> directions = {
        {step, 0, 0}, {-step, 0, 0}, {0, step, 0}, {0, -step, 0}, {0, 0, step}, {0, 0, -step}
    };

    for (const auto& direction : directions) {
        geometry_msgs::Pose neighbor = pose;
        neighbor.position.x += std::get<0>(direction);
        neighbor.position.y += std::get<1>(direction);
        neighbor.position.z += std::get<2>(direction);

        if (neighbor.position.x >= 0 && neighbor.position.x < 20 &&
            neighbor.position.y >= 0 && neighbor.position.y < 30 &&
            neighbor.position.z >= 0 && neighbor.position.z < 15 &&
            !isInCollision(neighbor)) {
            neighbors.push_back(neighbor);
        }
    }

    ROS_INFO("Found %zu valid neighbors for pose (%f, %f, %f)", 
             neighbors.size(), pose.position.x, pose.position.y, pose.position.z);

    return neighbors;
}

struct Node {
    geometry_msgs::Pose pose;
    double g_cost;  // Custo do comeco ate o no
    double h_cost;  // Custo heuristico 
    Node* parent;

    double f_cost() const {
        return g_cost + h_cost;
    }

    bool operator>(const Node& other) const {
        return f_cost() > other.f_cost();
    }
};

// Algoritmo AStar
std::vector<geometry_msgs::Pose> aStar(const geometry_msgs::Pose& start, const geometry_msgs::Pose& goal, const geometry_msgs::Pose* previous = nullptr) {
    ROS_INFO("A* search from (%f, %f, %f) to (%f, %f, %f)", 
             start.position.x, start.position.y, start.position.z,
             goal.position.x, goal.position.y, goal.position.z);

    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_list;
    std::unordered_map<std::string, Node> all_nodes;

    Node start_node = {start, 0.0, euclideanDistance(start, goal), nullptr};
    open_list.push(start_node);
    all_nodes[std::to_string(start.position.x) + "," + std::to_string(start.position.y) + "," + std::to_string(start.position.z)] = start_node;

    int iterations = 0;
    while (!open_list.empty()) {
        Node current = open_list.top();
        open_list.pop();

        if (euclideanDistance(current.pose, goal) < 0.5) {  // Increased threshold
            ROS_INFO("Goal reached after %d iterations", iterations);
            std::vector<geometry_msgs::Pose> path;
            Node* node = &all_nodes[std::to_string(current.pose.position.x) + "," + std::to_string(current.pose.position.y) + "," + std::to_string(current.pose.position.z)];
            while (node) {
                path.push_back(node->pose);
                node = node->parent;
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        std::vector<geometry_msgs::Pose> neighbors = findNeighbors(current.pose);
        for (const auto& neighbor_pose : neighbors) {
            // Ignorar o vizinho se for igual ao ponto anterior (para evitar o retorno)
            if (previous && euclideanDistance(neighbor_pose, *previous) < 0.1) {
                continue;
            }

            double tentative_g_cost = current.g_cost + euclideanDistance(current.pose, neighbor_pose);

            std::string key = std::to_string(neighbor_pose.position.x) + "," + std::to_string(neighbor_pose.position.y) + "," + std::to_string(neighbor_pose.position.z);
            if (all_nodes.find(key) == all_nodes.end() || tentative_g_cost < all_nodes[key].g_cost) {
                Node neighbor = {neighbor_pose, tentative_g_cost, euclideanDistance(neighbor_pose, goal), &all_nodes[std::to_string(current.pose.position.x) + "," + std::to_string(current.pose.position.y) + "," + std::to_string(current.pose.position.z)]};
                open_list.push(neighbor);
                all_nodes[key] = neighbor;
            }
        }

        iterations++;
        if (iterations % 1000 == 0) {
            ROS_INFO("A* search iteration %d", iterations);
        }
    }

    ROS_WARN("A* search failed to find a path");
    return std::vector<geometry_msgs::Pose>();
}

std::vector<geometry_msgs::Pose> findPathThroughWaypoints(const std::vector<geometry_msgs::Pose>& waypoints) {
    std::vector<geometry_msgs::Pose> full_path;

    ROS_INFO("Finding path through %zu waypoints", waypoints.size());

    for (size_t i = 0; i < waypoints.size() - 1; ++i) {
        ROS_INFO("Finding path from waypoint %zu (%f, %f, %f) to %zu (%f, %f, %f)", 
                 i, waypoints[i].position.x, waypoints[i].position.y, waypoints[i].position.z,
                 i+1, waypoints[i+1].position.x, waypoints[i+1].position.y, waypoints[i+1].position.z);
        
        std::vector<geometry_msgs::Pose> segment = aStar(waypoints[i], waypoints[i + 1]);

        if (segment.empty()) {
            ROS_WARN("No path found between waypoints %zu and %zu", i, i+1);
            return std::vector<geometry_msgs::Pose>();
        }

        ROS_INFO("Path segment found with %zu points", segment.size());

        // Adiciona o segmento ao caminho total, evitando duplicar o ponto inicial do segmento
        if (i == 0) {
            full_path.insert(full_path.end(), segment.begin(), segment.end());
        } else {
            full_path.insert(full_path.end(), segment.begin() + 1, segment.end());
        }
    }

    ROS_INFO("Full path found with %zu points", full_path.size());
    return full_path;
}

// Receber mensagem 
void plantsBedsCallback(const std_msgs::String::ConstPtr &msg) {
    indices.clear();  // Clear previous indices
    std::istringstream ss(msg->data);
    std::string word;
    ss >> word;
    int num;
    while (ss >> num) {
        indices.push_back(num);
    }
    ROS_INFO("Mensagem recebida: %s", msg->data.c_str());
    ROS_INFO("Indices a serem visitados: ");
    for (int i : indices) {
        ROS_INFO("%d", i);
    }

    indices_received = true;
    new_indices_received = true;
    trajectory_planned = false;
    trajectory_published = false;
    ROS_INFO("Received new indices. Ready to plan new trajectory.");
}

//Gerar os pontos de acordo com cada canteiro
void generateWaypoints() {
    std::vector<std::vector<double>> raw_waypoints = {
        {4.0, 6.0, 1.1}, {4.0, 6.0, 3.9}, {4.0, 6.0, 6.7}, {4.0, 13.5, 1.1}, {4.0, 13.5, 3.9},
        {4.0, 13.5, 6.7}, {4.0, 21.0, 1.1}, {4.0, 21.0, 3.9}, {4.0, 21.0, 6.7}, {10.0, 6.0, 1.1},
        {10.0, 6.0, 3.9}, {10.0, 6.0, 6.7}, {10.0, 13.5, 1.1}, {10.0, 13.5, 3.9}, {10.0, 13.5, 6.7},
        {10.0, 21.0, 1.1}, {10.0, 21.0, 3.9}, {10.0, 21.0, 6.7}, {16.0, 6.0, 1.1}, {16.0, 6.0, 3.9},
        {16.0, 6.0, 6.7}, {16.0, 13.5, 1.1}, {16.0, 13.5, 3.9}, {16.0, 13.5, 6.7}, {16.0, 21.0, 1.1},
        {16.0, 21.0, 3.9}, {16.0, 21.0, 6.7}
    };
    
    //Gerar dois pontos com x+2 e x-2
    for (int i : indices) {
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = raw_waypoints[i - 1][0]-2;
        pose.pose.position.y = raw_waypoints[i - 1][1];
        pose.pose.position.z = raw_waypoints[i - 1][2];
        pose.pose.orientation.w = 1.0;

        points.push_back(pose);

        pose.pose.position.x = raw_waypoints[i - 1][0]+2;
        pose.pose.position.y = raw_waypoints[i - 1][1];
        pose.pose.position.z = raw_waypoints[i - 1][2];
        pose.pose.orientation.z = 1.0;
        pose.pose.orientation.w = 0.0;

        points.push_back(pose);
    }

    ROS_INFO("Waypoints gerados e categorizados.");
}

// Algoritmo do vizinho mais proximo para ordenacao dos pontos
void nearestNeighborTraversal(std::vector<geometry_msgs::PoseStamped>& positions, geometry_msgs::PoseStamped& initial) {
    while (!positions.empty()) {
        double min_distance = 100000;
        size_t min_index = 0;

        for (size_t i = 0; i < positions.size(); ++i) {
            double dist = euclideanDistance(initial.pose, positions[i].pose);
            if (dist < min_distance) {
                min_distance = dist;
                min_index = i;
            }
        }

        initial = positions[min_index];
        trajectory.push_back(initial);
        positions.erase(positions.begin()+min_index);
    }
}

void initializeStartAndEndPoints() {
    start_point.pose.position.x = 1.0;
    start_point.pose.position.y = 1.0;
    start_point.pose.position.z = 1.3;
    start_point.pose.orientation.x = 0.0;
    start_point.pose.orientation.y = 0.0;
    start_point.pose.orientation.z = 0.0;
    start_point.pose.orientation.w = 1.0;  

    end_point.pose.position.x = 1.0;
    end_point.pose.position.y = 1.0;
    end_point.pose.position.z = 1.0;
    end_point.pose.orientation.x = 0.0;
    end_point.pose.orientation.y = 0.0;
    end_point.pose.orientation.z = 0.0;
    end_point.pose.orientation.w = 1.0;  
}

void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    current_pose = *msg;
}

//Ajustar rotacao de pontos
std::vector<geometry_msgs::PoseStamped> processPoses(const std::vector<geometry_msgs::PoseStamped>& poses) {
    std::vector<geometry_msgs::PoseStamped> processed_poses;

    for (const auto& pose : poses) {
        geometry_msgs::PoseStamped new_pose = pose;

        // Verifica se x é 6, 12 ou 18 e ajusta a orientação
        if ((new_pose.pose.position.x == 6 || new_pose.pose.position.x == 12 || new_pose.pose.position.x == 18) &&
            (new_pose.pose.position.y == 6.0 || new_pose.pose.position.y == 13.5 || new_pose.pose.position.y == 21.0) &&
            new_pose.pose.position.z < 7) {
            new_pose.pose.orientation.x = 0;
            new_pose.pose.orientation.y = 0;
            new_pose.pose.orientation.z = 1;
            new_pose.pose.orientation.w = 0;
        } else {
            new_pose.pose.orientation.x = 0;
            new_pose.pose.orientation.y = 0;
            new_pose.pose.orientation.z = 0;
            new_pose.pose.orientation.w = 1;
        }

        // Adiciona a nova pose ao vetor processado
        processed_poses.push_back(new_pose);
    }

    return processed_poses;
}

void publishTrajectory(ros::Publisher& trajectory_pub) {
    if (final_path.empty() || trajectory_published) {
        return;
    }

    trajectory_msgs::MultiDOFJointTrajectory trajectory_msg;
    trajectory_msg.header.stamp = ros::Time::now();
    trajectory_msg.header.frame_id = "world";
    trajectory_msg.joint_names.push_back("base_link");

    for (const auto& pose : final_path) {
        trajectory_msgs::MultiDOFJointTrajectoryPoint point;
        
        geometry_msgs::Transform transform;
        transform.translation.x = pose.pose.position.x;
        transform.translation.y = pose.pose.position.y;
        transform.translation.z = pose.pose.position.z;
        transform.rotation = pose.pose.orientation;

        point.transforms.push_back(transform);
        
        // Add zero velocity and acceleration
        geometry_msgs::Twist zero_twist;
        point.velocities.push_back(zero_twist);
        point.accelerations.push_back(zero_twist);

        trajectory_msg.points.push_back(point);
    }

    trajectory_pub.publish(trajectory_msg);
    ROS_INFO("Published trajectory with %zu points", final_path.size());
    
    trajectory_published = true;
}

void planTrajectory() {
    if (!indices_received) {
        ROS_WARN("Indices not received yet.");
        return;
    }

    generateWaypoints();
    
    ROS_INFO("Number of waypoints generated: %zu", points.size());
    if (points.empty()) {
        ROS_WARN("No waypoints were generated!");
        return;
    }

    trajectory.clear();  // Clear any existing trajectory
    trajectory.push_back(start_point);
    nearestNeighborTraversal(points, start_point);
    trajectory.push_back(end_point);

    ROS_INFO("Number of points in trajectory: %zu", trajectory.size());

    // Convert trajectory from PoseStamped to Pose
    std::vector<geometry_msgs::Pose> trajectory_poses;
    for (const auto& pose_stamped : trajectory) {
        trajectory_poses.push_back(pose_stamped.pose);
    }

    std::vector<geometry_msgs::Pose> path = findPathThroughWaypoints(trajectory_poses);

    ROS_INFO("Number of points in path: %zu", path.size());
    if (path.empty()) {
        ROS_WARN("No valid path found!");
        return;
    }

    // Convert path back to PoseStamped
    caminho.clear();
    for (const auto& pose : path) {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.pose = pose;
        caminho.push_back(pose_stamped);
    }

    final_path = processPoses(caminho);

    total_planned_distance = 0.0;  // Reset the total distance
    for (size_t i = 1; i < final_path.size(); ++i) {
        double segment_distance = euclideanDistance(final_path[i - 1].pose, final_path[i].pose);
        total_planned_distance += segment_distance;
        ROS_INFO("Segment %zu distance: %f", i, segment_distance);
    }

    trajectory_planned = true;
    ROS_INFO("Trajectory planned with %zu points.", final_path.size());
    ROS_INFO("Total planned distance: %f meters", total_planned_distance);
    ROS_INFO("First waypoint: (%f, %f, %f)", final_path[0].pose.position.x, final_path[0].pose.position.y, final_path[0].pose.position.z);
    ROS_INFO("Last waypoint: (%f, %f, %f)", final_path.back().pose.position.x, final_path.back().pose.position.y, final_path.back().pose.position.z);
}

void trajectoryExecutionCallback(const trajectory_msgs::MultiDOFJointTrajectory::ConstPtr& msg) {
    ROS_INFO("Received trajectory with %zu points", msg->points.size());
    ROS_INFO("First point: (%f, %f, %f)", 
             msg->points[0].transforms[0].translation.x,
             msg->points[0].transforms[0].translation.y,
             msg->points[0].transforms[0].translation.z);
    ROS_INFO("Last point: (%f, %f, %f)", 
             msg->points.back().transforms[0].translation.x,
             msg->points.back().transforms[0].translation.y,
             msg->points.back().transforms[0].translation.z);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "path_planner");
    ros::NodeHandle nh;

    initializeStartAndEndPoints();  // Initialize start and end points

    ros::Subscriber sub_indices = nh.subscribe("/red/plants_beds", 1000, plantsBedsCallback);
    ros::Subscriber sub_pose = nh.subscribe("/red/carrot/pose", 1000, poseCallback);
    ros::Publisher trajectory_pub = nh.advertise<trajectory_msgs::MultiDOFJointTrajectory>("/red/tracker/input_trajectory", 1000);
    ros::Subscriber sub_trajectory_execution = nh.subscribe("/red/tracker/input_trajectory", 1000, trajectoryExecutionCallback);

    ros::Rate rate(10);  // 10 Hz, adjust as needed

    while (ros::ok()) {
        ros::spinOnce();

        if (new_indices_received) {
            ROS_INFO("Planning trajectory...");
            planTrajectory();
            if (trajectory_planned) {
                new_indices_received = false;
                ROS_INFO("Trajectory planned. Ready to publish.");
            } else {
                ROS_WARN("Failed to plan trajectory.");
            }
        }

        if (trajectory_planned && !trajectory_published) {
            publishTrajectory(trajectory_pub);
            if (trajectory_published) {
                ROS_INFO("Trajectory published. Waiting for new indices.");
                indices_received = false;  // Reset for the next set of indices
            } else {
                ROS_WARN("Failed to publish trajectory.");
            }
        }

        rate.sleep();
    }

    return 0;
}
