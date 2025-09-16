#include<speed_adaptive_control/apf_controller.h>
#include<speed_adaptive_control/reference_selector.h>
#include<speed_adaptive_control/jacobian_calculator.h>
// #include<speed_adaptive_control/franka_interface.h>

class MPC_Governor
{
    public:
        MPC_Governor(ros::NodeHandle nh, APF_Controller& apf_controller,
         ReferenceSelector& reference_selector,
         JacobianCalculator& jacobian_calculator);
        ~MPC_Governor();

        ros::NodeHandle nh;
        APF_Controller *ac;
        ReferenceSelector *rf;
        JacobianCalculator *jc;
        //Franka_Interface *fi;

        void forward(std::array<double, 7>& query_q);
        void update();
        void roll_predict();
        void run();

        void update_kinematic();

        std::array<double, 7> *qr;
        std::array<double, 7> vel_cmd;
        std::array<double, 7> qv;

        std::array<double, 7> *cur_pos;
        std::array<double, 7> *cur_vel;

        std::array<double, 7> virtual_q;
        std::array<std::array<double, 7>, 10> virtual_traj;

    private:
        int rate = 100;
        double dt = 1 / double(rate);
};      