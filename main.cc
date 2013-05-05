#include <iostream>
#include <vector>
#include <string>

#include "opencv2/opencv.hpp"

// Transform Intel Training Samples to OpenCV Readable Training Samples
void transformIntelSampleToOpenCVSample(const string &samp, const string &vec)
{
	FILE* fpsamintel = fopen(samp.c_str(), "rb");
    if (fpsamintel)
    {
        printf("open fpsamintel error\n");
        exit(-1)
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
    if (fpsamopencv)
    {
        printf("open fpsamopencv error\n");
        exit(-1)
    }
    int count = N * 2;
    cout << N * 2 << endl;
    fwrite( &count, sizeof( count ), 1, fpsamopencv );
    int vecsize = 24 * 24;
    fwrite( &vecsize, sizeof( vecsize ), 1, fpsamopencv );
    short tmp = 0;
    fwrite( &tmp, sizeof( tmp ), 1, fpsamopencv );
    fwrite( &tmp, sizeof( tmp ), 1, fpsamopencv );
    uchar chartmp = 0;
    for (int i = 0; i < N; i++) {
        uchar* pImg = (uchar*)(pImgData + i*imgarea);
        Mat img = Mat(ah, aw, CV_8UC1, pImg);
        Mat img_small = Mat(24, 24, CV_8UC1);
        cv::resize(img, img_small, Size(24, 24), 0, 0, INTER_AREA);
        Mat img_small_flip = Mat(24, 24, CV_8UC1);
        cv::flip(img_small, img_small_flip, 1);
        
        fwrite( &chartmp, sizeof( chartmp ), 1, fpsamopencv );
        for (int r = 0; r < 24; r++) {
            for (int c = 0; c < 24; c++) {
                short pixel = (short)img_small.at<uchar>(r, c);
                fwrite( &pixel, sizeof( pixel ), 1, fpsamopencv );
            }
        }
        
        fwrite( &chartmp, sizeof( chartmp ), 1, fpsamopencv );
        for (int r = 0; r < 24; r++) {
            for (int c = 0; c < 24; c++) {
                short pixel = (short)img_small_flip.at<uchar>(r, c);
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
        exit(-1)
    }
    
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
    vecsize = 24 * 24;
    fwrite( &vecsize, sizeof( vecsize ), 1, fvec3 );
    tmp = 0;
    fwrite( &tmp, sizeof( tmp ), 1, fvec3 );
    fwrite( &tmp, sizeof( tmp ), 1, fvec3 );
    
    for (int i = 0; i < nvec1; i++) {
        char chartmp = 0;
        fread( &chartmp, sizeof( chartmp ), 1, fvec1 );
        fwrite( &chartmp, sizeof( chartmp ), 1, fvec3 );
        short data[24*24];
        fread(data, sizeof(short), 24*24, fvec1);
        fwrite(data, sizeof(short), 24*24, fvec3);
    }
    
    for (int i = 0; i < nvec2; i++) {
        char chartmp = 0;
        fread( &chartmp, sizeof( chartmp ), 1, fvec2 );
        fwrite( &chartmp, sizeof( chartmp ), 1, fvec3 );
        short data[24*24];
        fread(data, sizeof(short), 24*24, fvec2);
        fwrite(data, sizeof(short), 24*24, fvec3);
    }
    
    fclose(fvec1);
    fclose(fvec2);
    fclose(fvec3);
}

int main()
{
//   combile_vecs(argv[1], argv[2], argv[3]); return 0;
//   transformIntelSampleToOpenCvSample(argv[1], argv[2]); return 0;
}