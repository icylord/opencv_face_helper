#include <iostream>
#include <vector>
#include <string>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void extract_intel_samples(const string &samp)
{
    FILE* fpsamintel = fopen(samp.c_str(), "rb");
    if (fpsamintel == NULL)
    {
        printf("open fpsamintel error\n");
        exit(-1);
    }
    
    int head[4];
    fread(head, sizeof(int), 4, fpsamintel);
    int aw = head[0];   // 40 in current cropped database
    int ah = head[1];   // 40 in current cropped database
    int N = head[2];    // 12102 faces currently in the cropped database
    assert( N > 0 );
    int imgarea = aw * ah;
    uchar* pImgData = (uchar*)cvAlloc(sizeof(uchar)*imgarea * N);
    fread(pImgData, sizeof(uchar), imgarea*N, fpsamintel);
    fclose(fpsamintel);

    char out_name[128];
    for (int i = 0; i < N; i++) {
        uchar* pImg = (uchar*)(pImgData + i*imgarea);
        Mat img = Mat(ah, aw, CV_8UC1, pImg);
        sprintf(out_name, "intel24/%d.png", i);
        Mat im24 = Mat(24, 24, CV_8UC1);
        cv::resize(img, im24, Size(24, 24), 0, 0, cv::INTER_AREA);
        imwrite(out_name, im24);
    }
}

// Transform Intel Training Samples to OpenCV Readable Training Samples
void transform_IntelSample_To_OpenCVSample(const string &samp, const string &vec)
{
    cout << samp << endl;
	FILE* fpsamintel = fopen(samp.c_str(), "rb");
    if (fpsamintel == NULL)
    {
        printf("open fpsamintel error\n");
        exit(-1);
    }
    
	int head[4];
	fread(head, sizeof(int), 4, fpsamintel);
	int aw = head[0];	// 40 in current cropped database
	int ah = head[1];	// 40 in current cropped database
	int N = head[2];	// 12102 faces currently in the cropped database
	assert( N > 0 );
	int imgarea = aw * ah;
	uchar* pImgData = (uchar*)cvAlloc(sizeof(uchar)*imgarea * N);
	fread(pImgData, sizeof(uchar), imgarea*N, fpsamintel);
	fclose(fpsamintel);
    
	FILE* fpsamopencv = fopen(vec.c_str(), "wb");
    if (fpsamopencv == NULL)
    {
        printf("open fpsamopencv error\n");
        exit(-1);
    }
    int count = N * 2;
    cout << N * 2 << endl;
    fwrite( &count, sizeof( count ), 1, fpsamopencv );
    int vecsize = 40 * 40;
    fwrite( &vecsize, sizeof( vecsize ), 1, fpsamopencv );
    short tmp = 0;
    fwrite( &tmp, sizeof( tmp ), 1, fpsamopencv );
    fwrite( &tmp, sizeof( tmp ), 1, fpsamopencv );
    uchar chartmp = 0;
    for (int i = 0; i < N; i++) {
        uchar* pImg = (uchar*)(pImgData + i*imgarea);
        Mat img = Mat(ah, aw, CV_8UC1, pImg);
        // Mat img_small = Mat(24, 24, CV_8UC1);
        // cv::resize(img, img_small, Size(24, 24), 0, 0, INTER_AREA);
        // Mat img_small_flip = Mat(24, 24, CV_8UC1);
        Mat img_flip = img;
        cv::flip(img, img_flip, 1);
        
        fwrite( &chartmp, sizeof( chartmp ), 1, fpsamopencv );
        for (int r = 0; r < 40; r++) {
            for (int c = 0; c < 40; c++) {
                short pixel = (short)img.at<uchar>(r, c);
                fwrite( &pixel, sizeof( pixel ), 1, fpsamopencv );
            }
        }
        
        fwrite( &chartmp, sizeof( chartmp ), 1, fpsamopencv );
        for (int r = 0; r < 40; r++) {
            for (int c = 0; c < 40; c++) {
                short pixel = (short)img_flip.at<uchar>(r, c);
                fwrite( &pixel, sizeof( pixel ), 1, fpsamopencv );
            }
        }
    }
    
    fclose(fpsamopencv);
}

// Combine Tow OpenCV Training Samples into One
void combile_vecs(const string &vec1, const string &vec2, const string &vec3)
{
    FILE* fvec1 = fopen(vec1.c_str(), "rb");
    FILE* fvec2 = fopen(vec2.c_str(), "rb");
    FILE* fvec3 = fopen(vec3.c_str(), "wb");
    if (fvec1 == NULL || fvec2 == NULL || fvec3 == NULL)
    {
        printf("open vec file error\n");
        exit(-1);
    }
    
    const int image_size = 24;
    int nvec1;
    int nvec2;
    fread(&nvec1, sizeof(nvec1), 1, fvec1);
    fread(&nvec2, sizeof(nvec2), 1, fvec2);
    int nvec3 = nvec1 + nvec2;
    int vecsize;
    fread(&vecsize, sizeof(vecsize), 1, fvec1);
    fread(&vecsize, sizeof(vecsize), 1, fvec2);

    short tmp = 0;
    fread( &tmp, sizeof( tmp ), 1, fvec1 );
    fread( &tmp, sizeof( tmp ), 1, fvec2 );
    fread( &tmp, sizeof( tmp ), 1, fvec1 );
    fread( &tmp, sizeof( tmp ), 1, fvec2 );
    
    int count = nvec3;
    cout << count << endl;
    fwrite( &count, sizeof( count ), 1, fvec3 );
    vecsize = image_size * image_size;
    fwrite( &vecsize, sizeof( vecsize ), 1, fvec3 );
    tmp = 0;
    fwrite( &tmp, sizeof( tmp ), 1, fvec3 );
    fwrite( &tmp, sizeof( tmp ), 1, fvec3 );
    
    for (int i = 0; i < nvec1; i++) {
        char chartmp = 0;
        fread( &chartmp, sizeof( chartmp ), 1, fvec1 );
        fwrite( &chartmp, sizeof( chartmp ), 1, fvec3 );
        short data[image_size*image_size];
        fread(data, sizeof(short), image_size*image_size, fvec1);
        fwrite(data, sizeof(short), image_size*image_size, fvec3);
    }
    
    for (int i = 0; i < nvec2; i++) {
        char chartmp = 0;
        fread( &chartmp, sizeof( chartmp ), 1, fvec2 );
        fwrite( &chartmp, sizeof( chartmp ), 1, fvec3 );
        short data[image_size*image_size];
        fread(data, sizeof(short), image_size*image_size, fvec2);
        fwrite(data, sizeof(short), image_size*image_size, fvec3);
    }
    
    fclose(fvec1);
    fclose(fvec2);
    fclose(fvec3);
}

void combile_vecs(const string &vec)
{
    FILE* fvec = fopen(vec.c_str(), "wb");
    if (fvec == NULL)
    {
        printf("open vec file error\n");
        exit(-1);
    }
    
    int num = 12102;
    int nvec = num * 4;
    int vecsize = 24 * 24;

    short tmp = 0;    
    int count = nvec;
    fwrite( &count, sizeof( count ), 1, fvec );
    fwrite( &vecsize, sizeof( vecsize ), 1, fvec );
    tmp = 0;
    fwrite( &tmp, sizeof( tmp ), 1, fvec );
    fwrite( &tmp, sizeof( tmp ), 1, fvec );

    for (int i = 0; i < num; ++i)
    {
        char vecname[64];
        sprintf(vecname, "vec24/%d.vec", i);
        FILE* fvec1 = fopen(vecname, "rb");
        int nvec1;
        fread(&nvec1, sizeof(nvec1), 1, fvec1);
        fread(&vecsize, sizeof(vecsize), 1, fvec1);

        fread( &tmp, sizeof( tmp ), 1, fvec1 );
        fread( &tmp, sizeof( tmp ), 1, fvec1 );
        for (int i = 0; i < 5; i++) {
            char chartmp = 0;
            fread( &chartmp, sizeof( chartmp ), 1, fvec1 );
            fwrite( &chartmp, sizeof( chartmp ), 1, fvec );
            short data[24*24];
            fread(data, sizeof(short), 24*24, fvec1);
            fwrite(data, sizeof(short), 24*24, fvec);
        }
        fclose(fvec1);
    }
    
    fclose(fvec);
}

void show_vec_file(const string &vec)
{
    FILE *p = fopen(vec.c_str(), "rb");

    if (p == NULL)
    {
        printf("Open File %s error\n", vec.c_str());
        exit(-1);
    }
    const int image_size = 24;
    int vecsize;
    short tmp;

    int count;
    fread( &count, sizeof( count ), 1, p );
    fread( &vecsize, sizeof( vecsize ), 1, p );
    cout << count << endl;
    cout << vecsize << endl;
    fread( &tmp, sizeof( tmp ), 1, p );
    fread( &tmp, sizeof( tmp ), 1, p );
    
    for (int i = 0; i < count; i++) {
        char chartmp = 0;
        fread( &chartmp, sizeof( chartmp ), 1, p );
        short data[image_size*image_size];
        fread(&data, sizeof(short), image_size*image_size, p);
        
        Mat im = Mat(image_size, image_size, CV_8UC1);
        int idx = 0;
        for (int r = 0; r < image_size; r++) {
            for (int c = 0; c < image_size; c++) {
                im.at<uchar>(r, c) = (uchar)data[idx++];
            }
        }
        // char img_name[64];
        // sprintf(img_name, "flickr002245/%d.png", i);
        // imwrite(img_name, im);
        imshow("TEST", im);
        while (1) {
            if (waitKey(10) == 'q') {
                break;
            }
        }
    }
    
    fclose(p);
}

void package_img_to_vec(const string &conf, const int num_image, const string dir, const string &vec)
{
    FILE *p = fopen(vec.c_str(), "wb");
    FILE *pconf = fopen(conf.c_str(), "r");
    const int image_size = 24;
    int vecsize = image_size * image_size;
    short tmp;
    int count = num_image;
    fwrite( &count, sizeof( count ), 1, p );
    fwrite( &vecsize, sizeof( vecsize ), 1, p );
    cout << count << endl;
    cout << vecsize << endl;
    fwrite( &tmp, sizeof( tmp ), 1, p );
    fwrite( &tmp, sizeof( tmp ), 1, p );
    char img_name[512];
    for (int i = 0; i < count; i++) {
        fscanf(pconf, "%s", img_name);
        Mat im = imread(dir+img_name, 0);
        Mat small_im = Mat(image_size, image_size, im.type());
        resize(im, small_im, Size(image_size, image_size), 0, 0, INTER_AREA);
        char chartmp = 0;
        fwrite( &chartmp, sizeof( chartmp ), 1, p);
        short data[image_size*image_size];
                
        int idx = 0;
        for (int r = 0; r < image_size; r++) {
            for (int c = 0; c < image_size; c++) {
                data[idx++] = (short)small_im.at<uchar>(r, c);
            }
        }
        fwrite(data, sizeof(short), vecsize, p);
    }
    
    fclose(p);
    fclose(pconf);
}

void get_canny_and_sq(cv::Mat &im)
{
    cv:equalizeHist(im, im);
    cv::Mat temp = cv::Mat(24, 24, CV_8UC1);
    cv::Canny(im, temp, 0, 50, 3);
    cv::Mat im_sumcanny = cv::Mat(25, 25, CV_32SC1);
    cv::integral(temp, im_sumcanny);
    cv::Mat im_sum = cv::Mat(25, 25, CV_32SC1);
    cv::Mat im_sq_sum = cv::Mat(25, 25, CV_64FC1);
    cv::integral(im, im_sum, im_sq_sum);
    cv::Rect equ_rect;
    int x = cvRound(24*0.15);
    int y = cvRound(24*0.15);
    int width = cvRound(24*0.7);
    int height = cvRound(24*0.7);
    int canny_value = im_sumcanny.at<int>(y, x) + im_sumcanny.at<int>(y + height, x + width)
        - im_sumcanny.at<int>(y, x + width) - im_sumcanny.at<int>(y + height, x);
    double sq_value = im_sq_sum.at<double>(y, x) + im_sq_sum.at<double>(y + height, x + width)
        - im_sq_sum.at<double>(y, x + width) - im_sq_sum.at<double>(y + height, x);
    printf("%d %lf\n", canny_value, sq_value);
}

void analysis_face_sq(const string &vec)
{
    FILE *p = fopen(vec.c_str(), "rb");

    if (p == NULL)
    {
        printf("Open File %s error\n", vec.c_str());
        exit(-1);
    }
    const int image_size = 24;
    int vecsize;
    short tmp;

    int count;
    fread( &count, sizeof( count ), 1, p );
    fread( &vecsize, sizeof( vecsize ), 1, p );
    cout << count << endl;
    cout << vecsize << endl;
    fread( &tmp, sizeof( tmp ), 1, p );
    fread( &tmp, sizeof( tmp ), 1, p );
    
    for (int i = 0; i < count; i++) {
        char chartmp = 0;
        fread( &chartmp, sizeof( chartmp ), 1, p );
        short data[image_size*image_size];
        fread(&data, sizeof(short), image_size*image_size, p);
        
        Mat im = Mat(image_size, image_size, CV_8UC1);
        int idx = 0;
        for (int r = 0; r < image_size; r++) {
            for (int c = 0; c < image_size; c++) {
                im.at<uchar>(r, c) = (uchar)data[idx++];
            }
        }
        get_canny_and_sq(im);
    }
}

int main(int argc, char **argv)
{
    // combile_vecs(argv[1], argv[2], argv[3]); return 0;
  // transform_IntelSample_To_OpenCVSample(argv[1], argv[2]); return 0;
    // combile_vecs(string(argv[1])); return 0;
    // show_vec_file(string(argv[1])); return 0;
    // analysis_face_sq(string(argv[1])); return 0;
    // extract_intel_samples(string("facepos.samp")); return 0;
    package_img_to_vec("flickr00.txt", 36240, "/Volumes/ShengyinWu/ShengyinWu/study/papers/dataset/FLICKRFACE/images/faces/faces/00/faces/", "/Volumes/ShengyinWu/ShengyinWu/study/papers/dataset/FLICKRFACE/images/faces/faces/00/flickr00.vec"); return 0;
    // package_img_to_vec("face.22.txt", 11626, "/Volumes/ShengyinWu/ShengyinWu/study/papers/dataset/flickr_yushiqi/flickr-images96x96-and-vec/", "face.22_40x40_11626.vec");
    // package_img_to_vec("flickrDM+45.txt", 17641, "/Volumes/ShengyinWu/ShengyinWu/study/papers/dataset/FLICKRFACE/images/faces/faces/45/faces/", "flickrDMR+45+17641.vec");
    // package_img_to_vec("flickrDMR+67.txt", 3875, "/Volumes/ShengyinWu/ShengyinWu/study/papers/dataset/FLICKRFACE/images/faces/faces67/67/faces/", "flickrDMR+67+3875.vec");
    //package_img_to_vec("/Volumes/ShengyinWu/ShengyinWu/study/papers/dataset/FLICKRFACE/images/faces/faces/00+22+45/flickrDM+00+22+45.txt", 25663, "/Volumes/ShengyinWu/ShengyinWu/study/papers/dataset/FLICKRFACE/images/faces/faces/00+22+45/faces/", "flickrDM+00+22+45+25663.vec");
}