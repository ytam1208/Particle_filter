#include "mouse.hpp"

class Particle
{
public:
    int x;
    int y;
    float weight;
};

class Random_Particle_make
{
private:
    Particle random_particle;
    std::vector<Particle> random_particle_vector;

    std::random_device rd1, rd2;

public:
    std::vector<Particle> make_coordinate(int _Particle_count, int _MAX_MAT_RANGE)
    {
        int max_particle_num = _Particle_count;
        while(_Particle_count != 0)
        {
            std::mt19937 gen1(rd1());
            std::mt19937 gen2(rd2());
            std::uniform_int_distribution<int> dis(0, _MAX_MAT_RANGE -1);
            random_particle.x = dis(gen1);
            random_particle.y = dis(gen2);
            random_particle.weight = 1.0 / max_particle_num;

            random_particle_vector.push_back(random_particle);
            _Particle_count--;
        }
        return random_particle_vector;
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
};

class Map
{
private:
    Random_Particle_make random;
    Particle particle;

    int init_ROW = 1000;
    int init_COL = 1000;

public:
    cv::Mat src;
    std::vector<Particle> particle_vector;
    int Particle_count = 10000;  //파티클의 갯수 1000개

public:
    Map()
    {
        src = cv::Mat(init_ROW, init_COL, CV_8UC3, cv::Scalar(0, 0, 0));
    }
    ~Map()
    {}

    void initMap()
    {
        std::random_device rd1, rd2;
        std::uniform_int_distribution<int> dis(0, Particle_count - 1);

        if (src.channels() == 3)
        {
            int MAX_MAT_RANGE = ((init_ROW+init_COL) / 2);
            particle_vector = random.make_coordinate(Particle_count, MAX_MAT_RANGE);
        }

        for (int i = 0; i < Particle_count; i++)
            cv::circle(src, cv::Point(particle_vector[i].x, particle_vector[i].y), 2, cv::Scalar(0, 0, 255), 1, -1, 0);
    }

    std::vector<Particle> checkOutlier(std::vector<Particle> &_init_particle_vector)
    {
        for(int i = 0; i < Particle_count; i++)
        {
            if(_init_particle_vector[i].x <= 0 || _init_particle_vector[i].y <= 0)
                _init_particle_vector[i].weight = 0;
            else if(_init_particle_vector[i].x >= 1000 || _init_particle_vector[i].y >= 1000)
                _init_particle_vector[i].weight = 0;
        }
        return _init_particle_vector;
    }
};



class Motion
{
private:
    MouseInterface mouse_event;
    Particle motion_particle;
    Particle target_pose;

    Map first_map;
    Map check_map;
    Random_Particle_make random;

private:
    std::thread process;
    std::mutex RUN;

    cv::Mat init_map;
    cv::Mat copy_first_init_map;
    cv::Mat motion_map;
    cv::Point after_point;
    cv::Point click_point;

    float Particle_weight_Up = 2.0f;
    int Observation_range = 200;

private:
    int particle_count = first_map.Particle_count;
    std::vector<Particle> motion_particle_vector;

public:
    std::vector<Particle> init_particle_vector;

public:
    void motion_process()
    {
        char END_COMMAND;
        while (1 || (END_COMMAND = getchar()) != EOF)
        {
            init_map.setTo(cv::Scalar(0, 0, 0));

            for (int i = 0; i < particle_count; i++)
            {
                motion_particle.x = motion_particle_vector[i].x + (int)random.GaussianRandom();
                motion_particle.y = motion_particle_vector[i].y + (int)random.GaussianRandom();
                motion_particle.weight = motion_particle_vector[i].weight;
                init_particle_vector.push_back(motion_particle);
            }
            for (int i = 0; i < particle_count; i++)
                cv::circle(init_map, cv::Point(init_particle_vector[i].x, init_particle_vector[i].y), 1, cv::Scalar(0, 0, 127), 2, -1, 0);

            // cv::imshow("motion_map", init_map);
            init_particle_vector = check_map.checkOutlier(init_particle_vector);
            
            if (click_point.x > 0 && click_point.y > 0)
            {
                cv::circle(init_map, cv::Point(click_point.x, click_point.y), Observation_range, cv::Scalar(255, 0, 0), 1, -1, 0);
                cv::circle(init_map, cv::Point(click_point.x, click_point.y), 1, cv::Scalar(0, 255, 0), 4, -1, 0);
            }
            Circle_check(&click_point, init_particle_vector , init_map);

            init_particle_vector = Normalize_Particle_Weight(init_particle_vector);
            init_particle_vector = Resampling(init_particle_vector, init_map);
            
            cv::imshow("motion_map", init_map);
            cv::setMouseCallback("motion_map", MouseInterface::CallBackFunc, &click_point);
            cv::waitKey(1);
            init_particle_vector.clear();
        }
    }

    void Circle_check(cv::Point *click_point, std::vector<Particle> &_init_particle_vector, cv::Mat &init_map)
    {
        cv::Point *zero_point = (cv::Point *)click_point;
        int x = zero_point->x;
        int y = zero_point->y;

        int in_particle_count = 0;
        double particle_dis_mean = 0.0;
        double variance = 0.0;

        int inlier_count = 0;
        int outlier_count= 0;
        int out_count = 0;

        for(int i = 0; i < particle_count; i++)
        {
            int x_ = pow(_init_particle_vector[i].x - x, 2);
            int y_ = pow(_init_particle_vector[i].y - y, 2);
            int distance = sqrt( x_ + y_);

            if(distance < Observation_range)
            {
                inlier_count++;
                cv::circle(init_map, cv::Point(_init_particle_vector[i].x, _init_particle_vector[i].y), 1, cv::Scalar(255, 0, 0), 2, -1, 0);
                _init_particle_vector[i].weight = _init_particle_vector[i].weight * Particle_weight_Up;
            }
            else if(_init_particle_vector[i].weight > 0)
                outlier_count++;

            if(_init_particle_vector[i].weight == 0)
            {
                out_count++;
                cv::circle(init_map, cv::Point(_init_particle_vector[i].x, _init_particle_vector[i].y), 1, cv::Scalar(255, 255, 0), 2, -1, 0);
            }
        }
        cv::Point pt0(50, 100), pt1(150,100), pt2(250, 100), pt3(400, 100), pt4(500, 100), pt5(600, 100), pt6(50, 50), pt7(350, 50);
        cv::putText(init_map, "Inlier ", pt0, 2, 1.2, cv::Scalar::all(125));
        cv::putText(init_map, std::to_string(inlier_count), pt1, 2, 1.2, cv::Scalar::all(125));

        cv::putText(init_map, "Outlier ", pt2, 2, 1.2, cv::Scalar::all(125));
        cv::putText(init_map, std::to_string(outlier_count), pt3, 2, 1.2, cv::Scalar::all(125));

        cv::putText(init_map, "Out ", pt4, 2, 1.2, cv::Scalar::all(125));
        cv::putText(init_map, std::to_string(out_count), pt5, 2, 1.2, cv::Scalar::all(125));

        cv::putText(init_map, "Total Particle ", pt6, 2, 1.2, cv::Scalar::all(125));
        cv::putText(init_map, std::to_string(particle_count), pt7, 2, 1.2, cv::Scalar::all(125));
    }

    std::vector<Particle> Normalize_Particle_Weight(std::vector<Particle> &_init_particle_vector)
    {
        float total_weight = 0.0f;
        float total = 0.0f;

        for(int i = 0; i < particle_count; i++)
            if(_init_particle_vector[i].weight > 0.0f)
                total_weight += _init_particle_vector[i].weight;

        for(int j = 0; j < particle_count; j++)
            if(_init_particle_vector[j].weight > 0.0f)
            {
                _init_particle_vector[j].weight /= total_weight;
                // std::cout << "weight = " << _init_particle_vector[j].weight << std::endl;
                total += _init_particle_vector[j].weight;
            }
        // std::cout << "Normalize total Weight = " << total << std::endl;
        return _init_particle_vector;
    }

    std::vector<Particle> Resampling(std::vector<Particle> &_init_particle_vector, cv::Mat &init_map)
    {
        std::vector<Particle> inlier_particle_vector;
        target_pose.x = 0;
        target_pose.y = 0;
        target_pose.weight = 0.0f;
        
        int count = 0;

        //가장 높은 weight 저장
        for(int i = 0; i < particle_count; i++)
        {
            if(target_pose.weight > _init_particle_vector[i].weight)
                target_pose.weight = target_pose.weight;                
            else if(target_pose.weight < _init_particle_vector[i].weight)
                target_pose.weight = _init_particle_vector[i].weight;
        }

        //가장 높은 target의 weight와 동일한 _init_particle_vector 위치를 inlier vector에 저장
        for(int i = 0; i < particle_count; i++)
        {
            if(target_pose.weight == _init_particle_vector[i].weight)
                inlier_particle_vector.push_back(_init_particle_vector[i]);
        }

        //weight가 0인 파티클의 위치를 inlier vector에 저장된 순서대로 좌표값을 넣어준다.
        for(int i = 0; i < particle_count; i++)
        {
            if(_init_particle_vector[i].weight == 0.0f)
            {
                if(count >= inlier_particle_vector.size())
                {
                    count = 0;
                    std::cout << "reset count!" << std::endl;
                }
                else
                {
                    _init_particle_vector[i].x = inlier_particle_vector[count].x;
                    _init_particle_vector[i].y = inlier_particle_vector[count].y;
                    _init_particle_vector[i].weight = inlier_particle_vector[count].weight;
                    cv::circle(init_map, cv::Point(_init_particle_vector[i].x, _init_particle_vector[i].y), 1, cv::Scalar(255, 255, 255), 1, -1, 0);
                    count++;
                }
            }
        }

        return _init_particle_vector;
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
    {}
};
