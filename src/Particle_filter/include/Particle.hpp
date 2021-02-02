
class Particle
{
public:
    int x;
    int y;
    float weight;

    
};

class Map
{
private:
    int init_ROW = 1000;
    int init_COL = 1000;
    Particle particle;

public:
    cv::Mat src;
    int Particle_count = init_COL / 2;
    std::vector<Particle> particle_vector;

public:
    Map()
    {
        src = cv::Mat(init_ROW, init_COL, CV_8UC3, cv::Scalar(0, 0, 0));
    }
    ~Map()
    {
    }

    void initMap()
    {
        std::random_device rd1, rd2;
        std::uniform_int_distribution<int> dis(0, init_COL - 1);
        int count = Particle_count;

        if (src.channels() == 3)
            while (count != 0)
            {
                std::mt19937 gen1(rd1());
                std::mt19937 gen2(rd2());
                particle.x = dis(gen1);
                particle.y = dis(gen2);
                particle.weight = 0.0;
                particle_vector.push_back(particle);

                count--;
            }

        std::cout << "Particle_count = " << particle_vector.size() << std::endl;

        for (int i = 0; i < Particle_count; i++)
            cv::circle(src, cv::Point(particle_vector[i].x, particle_vector[i].y), 2, cv::Scalar(0, 0, 255), 1, -1, 0);
    }
};

class Motion
{
private:
    Particle motion_particle;
    Particle target_pose;

    Map first_map;
    cv::Mat init_map;
    cv::Mat motion_map;

    // mouseInterface mouse_event;
public:
    std::thread process;
    std::mutex RUN;
    cv::Point after_point;
    cv::Point click_point;

private:
    int particle_count = first_map.Particle_count;
    std::vector<Particle> motion_particle_vector;
    std::vector<Particle> init_particle_vector;

public:

    static void CallBackFunc(int event, int x, int y, int flags, void* userdata)
    {
        if (event == cv::EVENT_LBUTTONDOWN)
        {
            cv::Point *p = (cv::Point *)userdata;
            p->x = x;
            p->y = y;
            // std::cout << "왼쪽 마우스 버튼 클릭.. 좌표 = (" << x << ", " << y << ")" << std::endl;
        }
    }

    void motion_process()
    {
        std::random_device rd1, rd2;
        std::uniform_int_distribution<int> dis(-1, 1);
        while(1)
        {
            std::mt19937 gen1(rd1());
            std::mt19937 gen2(rd2());

            init_map.setTo(cv::Scalar(0, 0, 0));

            for (int i = 0; i < particle_count; i++)
            {
                motion_particle.x = motion_particle_vector[i].x + (int)GaussianRandom();
                motion_particle.y = motion_particle_vector[i].y + (int)GaussianRandom();
                motion_particle.weight = motion_particle_vector[i].weight;
                init_particle_vector.push_back(motion_particle);

                // std::cout << init_particle_vector[i].x << " " << init_particle_vector[i].y << std::endl;
            }

            for (int i = 0; i < particle_count; i++)
                cv::circle(init_map, cv::Point(init_particle_vector[i].x, init_particle_vector[i].y), 2, cv::Scalar(0, 0, 255), 1, -1, 0);

            cv::imshow("motion_map", init_map);
            // std::cout << "x = " << mouse_event.x_pos << " " << "y = " << mouse_event.y_pos << std::endl;
            cv::setMouseCallback("motion_map", Motion::CallBackFunc, &click_point);
            if(after_point != click_point)
            {
                after_point = click_point;
                std::cout << "x = " << click_point.x << " y = " << click_point.y << std::endl;
            }
            else
            {
            }
            
            init_particle_vector.clear();
            cv::waitKey(1);
        }
    }

    double GaussianRandom()
    {
        double v1, v2, s;

        do
        {
            v1 = 2 * ((double)rand() / RAND_MAX) - 1; //-1.0 ~ 1.0 까지의 값
            v2 = 2 * ((double)rand() / RAND_MAX) - 1; //-1.0 ~ 1.0 까지의 값

            s = v1 * v1 + v2 * v2;
        } while (s >= 1 || s == 0);

        s = sqrt((-2 * log(s)) / s);

        return v1 * s;
    }

    void runloop()
    {
        motion_process();
    }

public:
    Motion()
    {
        after_point.x = 0;
        after_point.y = 0;

        first_map.initMap();
        motion_particle_vector.swap(first_map.particle_vector);

        init_map = first_map.src.clone();
    }
    ~Motion()
    {
    }
};
