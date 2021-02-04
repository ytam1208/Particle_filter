class MouseInterface
{
public:
	static void CallBackFunc(int event, int x, int y, int flags, void *userdata)
	{
		if (event == cv::EVENT_LBUTTONDOWN)
		{
			cv::Point *p = (cv::Point *)userdata;
			p->x = x;
			p->y = y;
			// std::cout << "왼쪽 마우스 버튼 클릭.. 좌표 = (" << x << ", " << y << ")" << std::endl;
		}		
	}
};
