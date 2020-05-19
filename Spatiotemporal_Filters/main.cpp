// Authors: Julian Parsert, Florian Meßner

#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>

void write_video(std::vector<cv::Mat> video, std::string filename, bool is_color=false) {
    cv::VideoWriter writer{filename, CV_FOURCC('M', 'J', 'P', 'G'), 24, video[0].size(), true};
    for (auto& frame : video) {
        if (!is_color)
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        writer << frame;
    }
}

std::vector<cv::Mat> to_spatio_temporal(cv::VideoCapture& video) {
    std::vector<cv::Mat> spatio_temporal;
    bool reading = true;
    do {
        cv::Mat frame;
        if (video.read(frame)) {
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
            spatio_temporal.push_back(frame);
        }
        else {
            reading = false;
        }
    } while(reading);
    return spatio_temporal;
}

std::vector<cv::Mat> convert_to_over_X( const std::vector<cv::Mat>& spatio_temporal) {
    int x_size = spatio_temporal[0].cols;
    std::vector<cv::Mat> over_x;
    for(int x = 0; x < x_size ; ++x) {
        cv::Mat tmp;
        for(const auto& fr : spatio_temporal) {
            cv::Mat t1;
            cv::transpose(fr.col(x) , t1);
            tmp.push_back(t1); // PUSHBACK only adds to the first row not to the collumn
        }
        cv::Mat t1;
        cv::transpose(tmp , t1);
        over_x.push_back(tmp);
    }
    return over_x;
}

std::vector<cv::Mat> convert_to_over_Y( const std::vector<cv::Mat>& spatio_temporal) {
    int y_size = spatio_temporal[0].rows;
    std::vector<cv::Mat> over_y;
    for(int y = 0; y < y_size ; ++y) {
        cv::Mat tmp;
        for (const auto &fr : spatio_temporal) {
            tmp.push_back(fr.row(y));
        }

        over_y.push_back(tmp);
    }
    return over_y;
}

std::vector<cv::Mat> get_conv_gabor_kernels(const std::initializer_list<double>& t) {
    cv::Size gabor_size={9,9};
    //init Gabor Kernel
    std::vector<cv::Mat> conv_gab_kernel;
    for (const double& d : t) {
        cv::Mat kernel = cv::getGaborKernel(gabor_size, 1 /*sigma*/,d /*theta*/, 1 /*lambd*/, 2 /*gamma*/, CV_PI*0.5 /*psi*/, CV_32F /*ktype*/);
        cv::flip(kernel, kernel, 1);
        conv_gab_kernel.push_back(kernel);
    }
    return conv_gab_kernel;
}

std::vector<cv::Mat> convolve_with_9_tap (const std::vector<cv::Mat>& over_y) {
    float in_x[9] = {0.0094, 0.1148, 0.3964, -0.0601, -0.9213, -0.0601, 0.3964, 0.1148,0.0094};
    float in_t[9] = {0.0008, 0.0176, 0.1660, 0.6383, 1.0, 0.6383, 0.1660, 0.0176, 0.0008};
    cv::Mat x_axis_filter(1, 9 , CV_32F, in_x);
    cv::Mat y_axis_filter(1, 9, CV_32F, in_t); //Maybe 9,1 maybe 1,9
    std::vector<cv::Mat> result;
    for(auto frame : over_y) {
        cv::Mat tmp;
        //frame.convertTo(frame , CV_32F);
        //normalize(frame , frame, 0 ,255.0, cv::NORM_MINMAX);
        cv::sepFilter2D(frame , tmp, CV_32F, x_axis_filter, y_axis_filter );
        cv::pow(tmp, 2, tmp);
        normalize(tmp , tmp, 0 ,255.0, cv::NORM_MINMAX);
        tmp.convertTo(tmp , CV_8UC1);
        result.push_back(tmp);
    }
    return result;
}

std::vector<std::vector<cv::Mat>> get_energy_of_gabor_in_spatio_temp(const std::vector<std::vector<cv::Mat>>& volumes , const std::vector<cv::Mat>& conv_gab_kernel) {
    std::vector<std::vector <cv::Mat>> gabor_videos;
    for (auto& volume : volumes) {
        std::vector <cv::Mat> gabor_video;
        // to get the gabor energy
        for (auto &x : volume) {
            cv::Mat gabor;
            for (unsigned int i = 0; i < conv_gab_kernel.size(); ++i) {
                cv::Mat tmp;
                filter2D(x,tmp,CV_32F, conv_gab_kernel[i], cv::Point(-1,-1));
                //cv::normalize(tmp, tmp , 0 , 255 , cv::NORM_MINMAX);
                // square the gabor
                cv::multiply(tmp, tmp, tmp);
                // sum them up
                if (i == 0) {
                    gabor = tmp.clone();
                } else {
                    cv::add(gabor, tmp, gabor);
                }
            }
            // take square root
            cv::pow(gabor, 0.5, gabor);
            // normalize
            normalize(gabor,gabor,0,255.0, cv::NORM_MINMAX);
            gabor.convertTo(gabor, CV_8UC1);
            gabor_video.push_back(gabor);
        }
        // write energy of gabor video
        //write_video(gabor_video, "gabor-energy-" + std::to_string(i) + ".avi");
        gabor_videos.push_back(gabor_video);
    }
    return gabor_videos;
}

int main(int argc, char** argv) {
    std::string input = "pen.mp4";
    if (argc > 1)
        input = argv[1];

    cv::VideoCapture video{input};
    if (!video.isOpened()){
        throw std::exception();
    }
    std::cout << "Reading video " << std::endl;
    auto spatio_temporal = to_spatio_temporal(video);
    video.release();

    std::cout << "Preparing spatio-temporal volumes" << std::endl;
    std::vector<cv::Mat> over_x = convert_to_over_X(spatio_temporal);
    std::vector<cv::Mat> over_y = convert_to_over_Y(spatio_temporal);
    // collect x-t and y-t volume in vector
    std::vector< std::vector <cv::Mat> > volumes{{over_x, over_y}};


    std::cout << "Calculating  Gabor Kernels" << std::endl;
    //Get Gabor kernels
    std::vector<cv::Mat> conv_gab_kernel = get_conv_gabor_kernels({M_PI/4, (3*M_PI) / 4});

    std::cout << "Calculating Energy of Gabor" << std::endl;
    // iterate over both
    std::vector<std::vector <cv::Mat>> gabor_videos = get_energy_of_gabor_in_spatio_temp(volumes , conv_gab_kernel);

    std::cout << "Calculating back from x-t to x-y" << std::endl;
    // rotate cube from x-t back to get x-y slices again
    std::vector<cv::Mat> back_to_xy;
    for(int y = 0; y < gabor_videos[1][0].rows; ++y) {
        cv::Mat tmp;
        for (const auto &fr : gabor_videos[1]) {
            tmp.push_back(fr.row(y));
        }
        back_to_xy.push_back(tmp);
    }
    std::cout << "Writing video Energy x-t " << std::endl;
    write_video(back_to_xy, "gabor-energy-x-t.avi");

    std::cout << "Applying 9-Tap filters" << std::endl;
    auto nine_tap = convolve_with_9_tap(over_y);
    back_to_xy.clear();
    for(int y = 0; y < nine_tap[0].rows; ++y) {
        cv::Mat tmp;
        for (const auto &fr : nine_tap) {
            tmp.push_back(fr.row(y));
        }
        back_to_xy.push_back(tmp);
    }
    std::cout << "Writing video Energy x-t " << std::endl;
    write_video(back_to_xy, "9-Tap-x-t.avi");

    return 0;
}
