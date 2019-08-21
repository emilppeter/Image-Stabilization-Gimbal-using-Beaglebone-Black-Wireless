#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include<opencv2/opencv.hpp>   // C++ OpenCV include file
#include <iostream>
#include <vector>
#include <cmath>

#define CAPTURE 0
#define WIDTH 1280
#define HEIGHT 720
#define PI 3.1415926535897932384626433
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    Mat frame, frame_HSV, frame_threshold,src,canny_output;
    int base, opposite_side,count=0;
    float roll_error,radius;
    struct timespec start, end;
    uint64_t diff;
    Point midpoint,error,center1,center2;
    Point2f center_circle;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    #if CAPTURE
    VideoCapture capture(0);   // capturing from /dev/video0

    cout << "Started Processing - Capturing Image" << endl;
    // set any  properties in the VideoCapture object
    capture.set(CV_CAP_PROP_FRAME_WIDTH,WIDTH);   // width pixels
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,HEIGHT);   // height pixels
    capture.set(CV_CAP_PROP_GAIN, 0);            // Enable auto gain etc.
    if(!capture.isOpened()){   // connect to the camera
       cout << "Failed to connect to the camera." << endl;
    }

    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);

    capture >> src;          // capture the image to the frame
    if(src.empty()){         // did the capture succeed?
       cout << "Failed to capture an image" << endl;
       return -1;
    }
    cout << "Processing - Performing Image Processing" << endl;
    #else
    src = imread("../RohitProject3/Input_image.jpg", CV_LOAD_IMAGE_COLOR); //reading image from jpeg file
    cout << "Reading Image from file" << endl;
    #endif

    imwrite( "../RohitProject3/Input_image.jpg", src );                     //saving source image

    // Blurring the image using gaussian blur
    GaussianBlur( src, frame, Size( 9, 9 ), 0, 0 );                         
 	imwrite( "../RohitProject3/Blur_image.jpg", frame );                      //saving blur image

    // Convert from BGR to HSV colorspace
    cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
    imwrite( "../RohitProject3/HSV_image.jpg", frame_HSV );                     //saving image in hsv format

    // Detect the object based on HSV Range Values
    inRange(frame_HSV, Scalar(20, 100, 100), Scalar(50, 255, 255), frame_threshold);
    imwrite( "../RohitProject3/Threshold_image.jpg", frame_threshold );                //saving image after applying threshold
    
    // Detect edges using canny
    Canny( frame_threshold, canny_output, 50, 150, 3 );
    imwrite( "../RohitProject3/Edges_image.jpg", canny_output );                    //saving edges of image
 	
    // Find contours
    findContours( canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );
 
    // Get the moments
    vector<Moments> mu(contours.size());
    for( int i = 0; i<contours.size(); i++ )
    { mu[i] = moments( contours[i], false ); }
 
    // Get the centroid of figures.
    vector<Point2f> mc(contours.size());
    for( int i = 0; i<contours.size(); i++)
    { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }
 
    // Draw contours
    Mat drawing(canny_output.size(), CV_8UC3, Scalar(255,255,255));
    vector<Point> approx_triangle; 
    for( int i = 0; i<contours.size(); i++ )
    {
        Scalar color = Scalar(167,151,0); // B G R values
        approxPolyDP(contours[i], approx_triangle, 0.05*arcLength(contours[i],true), true);     //approximating polygons obatined from contours 
        if (approx_triangle.size()==3)                                                          //detecting triangles among contours detected
        {
        	drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());             //drawing detected traingles
        	minEnclosingCircle(contours[i], center_circle, radius);                            //encircling detected triangles
        	circle( drawing, center_circle, radius, color, 1, 8, 0 );                          //drawing circles at the center of each triangle
            if (count==0)
                center1=mc[i];
            else if (count==1)
                center2=mc[i];
        	count++;
    	}
    }
    //printf("Count :%d\n",count );
    
    Point center(WIDTH/2,HEIGHT/2);                                             //finding center of image
    circle(drawing, center, 5, Scalar(128,0,0), 1);                     
    line(drawing, center1, center2, Scalar(128,0,0), 1, 8, 0 );                 //drawing line from center of one triangle to that of another
    
    // Finding midpoint of line  
    midpoint.x=(center1.x+center2.x)/2;                                         
    midpoint.y=(center1.y+center2.y)/2;
    circle(drawing, midpoint, 5, Scalar(128,0,0), 1);
   
    error.x=(midpoint.x-center.x);                                              //Calculating pan error
    error.y=(midpoint.y-center.y);                                              //Calculating tilt error
    
    base=center1.x-center2.x;                                                   
    opposite_side=center1.y-center2.y;

    //Calculating roll error
    if (abs(error.x)<abs(error.y))
        roll_error=atanf((float)opposite_side/base);                            
    else 
        roll_error=atanf((float)base/opposite_side);
    
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);

    //Calculating time taken
    diff = 1000000000 * (end.tv_sec - start.tv_sec) +
        end.tv_nsec - start.tv_nsec;

    printf("Pan error: %d Tilt error: %d Roll error: %lf\n",error.x,error.y,roll_error*(180/PI) );
    printf("Time taken: %10.3f ms\n",(diff/1000000.0) );
    
    //Saving final image
    imwrite( "../RohitProject3/Output.jpg", drawing );
    return 0;
}