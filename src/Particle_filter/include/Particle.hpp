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
    cv::RNG rng;
    Particle random_particle;
    std::vector<Particle> random_particle_vector;

    std::random_device rd1, rd2;

public:
    std::vector<Particle> make_coordinate(int _Particle_count, int _MAX_MAT_RANGE)
    {
        int max_particle_num = _Particle_count;
        while (_Particle_count != 0)
        {
            std::mt19937 gen1(rd1());
            std::mt19937 gen2(rd2());
            std::uniform_int_distribution<int> dis(0, _MAX_MAT_RANGE - 1);
            random_particle.x = dis(gen1);
            random_particle.y = dis(gen2);
            // random_particle.weight = 1.0 / max_particle_num;

            random_particle_vector.push_back(random_particle);
            _Particle_count--;
        }
        return random_particle_vector;
    }

    double GaussianRandom() //make Gaussian normal distribution
    {
        int average = 0;
        double segma = 10.0;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(average, segma);
        double result = std::round(dist(gen));
        
        return result;
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
    int Particle_count = 1000; //총 생성 파티클의 갯수

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
        if (src.channels() == 3)
        {
            int MAX_MAT_RANGE = ((init_ROW + init_COL) / 2);
            particle_vector = random.make_coordinate(Particle_count, MAX_MAT_RANGE);
        }

        for (int i = 0; i < Particle_count; i++)
            cv::circle(src, cv::Point(particle_vector[i].x, particle_vector[i].y), 2, cv::Scalar(0, 0, 255), 1, -1, 0);
    }

    std::vector<Particle> checkOutlier(std::vector<Particle> &_init_particle_vector)
    {
        for (int i = 0; i < Particle_count; i++)
        {
            if (_init_particle_vector[i].x <= 0 || _init_particle_vector[i].y <= 0)
                _init_particle_vector[i].weight = 0.0f;
            else if (_init_particle_vector[i].x >= init_COL || _init_particle_vector[i].y >= init_COL)
                _init_particle_vector[i].weight = 0.0f;
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
    float Particle_weight_Down = 1.0f;

    int Observation_range = 200;

private:
    int particle_count = first_map.Particle_count;
    std::vector<Particle> motion_particle_vector;
    std::vector<Particle> Sampling_vector;

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

                motion_particle.weight = 1.0f / (float)particle_count;
                init_particle_vector.push_back(motion_particle);
            }
            for (int i = 0; i < particle_count; i++)
                cv::circle(init_map, cv::Point(init_particle_vector[i].x, init_particle_vector[i].y), 1, cv::Scalar(0, 0, 127), 2, -1, 0);

            if (mouse_event.click_flag)
            {
                if (click_point.x > 0 && click_point.y > 0)
                {
                    cv::circle(init_map, cv::Point(click_point.x, click_point.y), Observation_range, cv::Scalar(255, 0, 0), 1, -1, 0);
                    cv::circle(init_map, cv::Point(click_point.x, click_point.y), 1, cv::Scalar(0, 255, 0), 4, -1, 0);
                }

                Circle_check(&click_point, init_particle_vector, init_map);
                init_particle_vector = check_map.checkOutlier(init_particle_vector);
                init_particle_vector = Normalize_Particle_Weight(init_particle_vector);
                Uniform_Resampling(&click_point, init_particle_vector, init_map);

                motion_particle_vector.swap(Sampling_vector);
                Sampling_vector.clear();
            }
            else
                motion_particle_vector.swap(init_particle_vector);

            cv::imshow("motion_map", init_map);
            cv::setMouseCallback("motion_map", MouseInterface::CallBackFunc, &click_point);
            cv::waitKey(1);

            if (click_point.x > 0 && click_point.y > 0)
                mouse_event.click_flag = true;

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
        int outlier_count = 0;
        int out_count = 0;

        for (int i = 0; i < particle_count; i++)
        {
            int x_ = pow(_init_particle_vector[i].x - x, 2);
            int y_ = pow(_init_particle_vector[i].y - y, 2);
            int distance = sqrt(x_ + y_);

            if (distance < Observation_range)
            {
                inlier_count++;
                cv::circle(init_map, cv::Point(_init_particle_vector[i].x, _init_particle_vector[i].y), 1, cv::Scalar(255, 0, 0), 2, -1, 0);
                _init_particle_vector[i].weight = _init_particle_vector[i].weight * Particle_weight_Up;
            }
            else
            {
                outlier_count++;
                _init_particle_vector[i].weight = _init_particle_vector[i].weight * Particle_weight_Down;
            }
        }
        cv::Point pt0(50, 100), pt1(150, 100), pt2(290, 100), pt3(420, 100), pt6(50, 50), pt7(350, 50);
        cv::putText(init_map, "Inlier ", pt0, 2, 1.2, cv::Scalar::all(125));
        cv::putText(init_map, std::to_string(inlier_count), pt1, 2, 1.2, cv::Scalar::all(125));

        cv::putText(init_map, "Outlier ", pt2, 2, 1.2, cv::Scalar::all(125));
        cv::putText(init_map, std::to_string(outlier_count), pt3, 2, 1.2, cv::Scalar::all(125));

        cv::putText(init_map, "Total Particle ", pt6, 2, 1.2, cv::Scalar::all(125));
        cv::putText(init_map, std::to_string(particle_count), pt7, 2, 1.2, cv::Scalar::all(125));
    }

    std::vector<Particle> Normalize_Particle_Weight(std::vector<Particle> &_init_particle_vector)
    {
        float total_weight = 0.0f;
        float total = 0.0f;

        for (int i = 0; i < particle_count; i++)
            if (_init_particle_vector[i].weight > 0.0f)
                total_weight += _init_particle_vector[i].weight;

        for (int j = 0; j < particle_count; j++)
            if (_init_particle_vector[j].weight > 0.0f)
            {
                _init_particle_vector[j].weight /= total_weight;
                // std::cout << "weight = " << _init_particle_vector[j].weight << std::endl;
                total += _init_particle_vector[j].weight;
            }
        return _init_particle_vector;
    }

    void Uniform_Resampling(cv::Point *click_point, std::vector<Particle> &_init_particle_vector, cv::Mat &init_map)
    {
        cv::Point *zero_point = (cv::Point *)click_point;
        int x = zero_point->x;
        int y = zero_point->y;

        float Search_weight = 0.0f;
        float add_particle_weight = 0.0f;
        Sampling_vector.reserve(_init_particle_vector.size());

        std::random_device rd;
        std::default_random_engine gen(rd());
        std::uniform_real_distribution<float> dist(0.0, (float)(1.0 / particle_count));

        for (int i = 0; i < particle_count; i++)
        {
            add_particle_weight += _init_particle_vector[i].weight;
            if (i == 0)
                Search_weight += dist(gen);

            if (Search_weight > add_particle_weight)
            {
            }

            else if (Search_weight <= add_particle_weight)
            {
                if (i != 0)
                    Search_weight += 1.0 / particle_count;

                Particle Sample;
                Sample.x = _init_particle_vector[i].x;
                Sample.y = _init_particle_vector[i].y;
                Sample.weight = _init_particle_vector[i].weight;
                Sampling_vector.push_back(Sample);

                add_particle_weight = 0.0f;
            }
            if (i == particle_count - 1)
                i = 0;
            if (Sampling_vector.size() == _init_particle_vector.size())
                break;
        }
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
