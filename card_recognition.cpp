/*
 * card_recognition.cpp
 *
 * CARD RECOGNITION
 *
 * Diogo Ferreira - Pedro Martins - 2017
 */


#include <iostream>

#include <math.h>

#include <map>

#include <stdlib.h>

#include <dirent.h>

#include "opencv2/core/core.hpp"

#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/highgui/highgui.hpp"

#include "opencv2/core/mat.hpp"

using namespace std;
using namespace cv;

void preProcessImage(Mat &originalImage)
{
    // Convert to a single-channel, intensity image
    if (originalImage.channels() > 1)
	    cvtColor(originalImage, originalImage, COLOR_BGR2GRAY, 1);

    // Apply Gaussian Filter to get only the countour of the symbols and
    // remove all the artifacts
    int n = 3;
    double sigmaX = 100.0;
    GaussianBlur(originalImage, originalImage, Size(n, n), sigmaX);

    // Apply threshold
    int thresholdValue = 150;
    int thresholdType = THRESH_BINARY;
    int maxValue = 255;
    threshold( originalImage, originalImage, 
            thresholdValue, maxValue, thresholdType);
}

// From: https://stackoverflow.com/questions/13495207/opencv-c-sorting-contours-by-their-contourarea
bool reverseCompareContourArea(vector<Point> c1, 
        vector<Point> c2)
{
    return contourArea(c1, false) > contourArea(c2, false);
}

// Adapted from: https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
void orderPointsCC(const vector<Point2f> &points, Point2f* orderedPoints) 
{
    int sMax = 0, sMin = 0, dMax = 0, dMin = 0;
    
    // Bottom-left will have the smallest sum
    // Top-right will have the bigger sum
    for (int i = 0; i < points.size(); i++) {
        double sum = points[i].x + points[i].y;
        if (sum > (points[sMax].x + points[sMax].y)) 
            sMax = i;
        if (sum < (points[sMin].x + points[sMin].y)) 
            sMin = i;
    }
    
    // Bottom-right will have the smallest difference
    // Top-left will have the bigger difference
    for (int i = 0; i < points.size(); i++) {
        if (i == sMax or i == sMin) 
            continue;

        double diff = points[i].x - points[i].y;
         
        if (diff < (points[dMax].x - points[dMax].y)) 
            dMin = i;
        if (diff > (points[dMin].x - points[dMin].y)) 
            dMax = i;
    }
    
    orderedPoints[0] = points[sMin];
    orderedPoints[1] = points[dMax];
    orderedPoints[2] = points[sMax];
    orderedPoints[3] = points[dMin];
}

void findContours(const Mat &image, int numCards,
        vector<vector<Point> > &cardsContours)
{
    //vector<Vec4i> hierarchy;
    Mat cannyOutput;
    int mode = CV_RETR_TREE;
    int method = CV_CHAIN_APPROX_SIMPLE;

    Canny(image, cannyOutput, 120, 240);

    findContours(cannyOutput, cardsContours, mode, method, Point(0, 0));

    // Find the most common contours
    sort(cardsContours.begin(), cardsContours.end(), reverseCompareContourArea);
    cardsContours.resize(numCards);
}

void transformCardContours(const Mat &image, vector<Mat> &cards,
        const vector<vector<Point> > &cardsContours)
{
    vector<Point> points;

    // Transform perspective card into a 500x500 image card
    for (int i = 0; i < cardsContours.size(); i++) {
        
        Mat card;

        points = cardsContours[i];
        
        // Compute approximation accuracy
        double epsilon = 0.02 * arcLength(points, true);
        vector<Point> approxCurve;
        approxPolyDP(points, approxCurve, epsilon, true);
        // Get rectangle of the minimum area
        RotatedRect boxRect = minAreaRect(approxCurve);
        
        /* 
        // Debug rect
        Mat drawing = image.clone();
        Scalar color = Scalar(128, 128, 128);
        // contour
        //drawContours( drawing, cardsContours, i, color, 5, 8, vector<Vec4i>(), 0, Point() );
        Point2f rect_points[4]; boxRect.points( rect_points );
        for( int j = 0; j < 4; j++ )
          line( drawing, rect_points[j], rect_points[(j+1)%4], color, 5, 8 );
         
        /// Show in a window
        namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
        imshow( "Contours", drawing ); 
        */
        
        // Correct order 
        //  3--2
        //  |  |
        //  0--1
        
        Point2f srcVertices[4];
        Point2f dstVertices[4];
        
        // Vertices of the rect box 
        dstVertices[0] = Point2f(0, 0);
        dstVertices[1] = Point2f(449, 0); 
        dstVertices[2] = Point2f(449, 449);        
        dstVertices[3] = Point2f(0, 449);        
        
        boxRect.points(srcVertices); 
        
        // Order vertices counter clockwise
        vector<Point2f> vertices(begin(srcVertices), end(srcVertices));
        orderPointsCC(vertices, srcVertices);
        
        // Get the transformation matrix
        Mat transform = getPerspectiveTransform(srcVertices, dstVertices);
        // Perform the transformation
        warpPerspective(image, card, transform, Size(450, 450));
        
        imwrite("rotated.jpg", card);
        
        cards.push_back(card);
    }
}

void getTrainingSet(const string path, 
        map<string, Mat> &cardDataset)
{
    DIR* dirp = opendir(path.c_str());
    struct dirent * dp;

    // Read all images in the training folder
    while ((dp = readdir(dirp)) != NULL) {
        string filename = dp->d_name;
        if (filename != "." and filename != "..") {
            filename = path + filename; 
            
            Mat image = imread(filename);
            
            // Pre-process image and save in dataset 
            preProcessImage(image);
            cardDataset[filename] = image;
        }
    }
}

int countBinaryWhite(Mat card) 
{
    int count = 0;
    for (int i = 0; i < card.rows; i++) {
        for (int j = 0; j < card.cols; j++) {
            if (card.at<uchar>(i, j) == 255)
                count++;
        }
    }
    return count;
}

void getClosestCard(Mat &card, map<string, Mat> &cards, 
        String &cardName)
{
    int i = -1;
    int diff, tmpDiff;
    map<string, Mat>::iterator it = cards.begin();

    while(it != cards.end()) {
        Mat compare_card = it->second;
        
        if (compare_card.size() != Size(450, 450))
            // Resize image to allow diff calculations 
            resize(compare_card, compare_card, Size(450, 450));

        /*
        // Display difference
        namedWindow( "Imagem Original"+i, WINDOW_AUTOSIZE );
	    imshow("Imagem Original"+i, abs(compare_card - card));
        */

        if (!++i){
            diff = countBinaryWhite(abs(compare_card - card));
            cardName = it->first;
        } else {
            tmpDiff = countBinaryWhite(abs(compare_card - card));
            if (tmpDiff < diff) {
                diff = tmpDiff;
                cardName = it->first;
            }
        }
        it++;
    }


}

int main( int argc, char** argv )
{
    
    if( argc != 2 )
    {
		cout << "The name of the image file is missing !!" << endl;

        return -1;
    }

    
    // Get dataset
    int numCards = 1;
    map<string, Mat> cardset; 
    getTrainingSet("./training_set/", cardset);

    Mat originalCard;

	originalCard = imread( argv[1], IMREAD_UNCHANGED );

	if( originalCard.empty() )
	{
		// NOT SUCCESSFUL : the data attribute is empty
		cout << "Image file could not be open !!" << endl;
	    return -1;
	}
    
    preProcessImage(originalCard);
    
    /* 
    // Display transformation
    namedWindow("Transformed", WINDOW_AUTOSIZE );
    imshow("Transformed", originalCard);
    */

    vector<vector<Point> > cardsContours;
    findContours(originalCard, numCards, cardsContours);
    vector<Mat> cards;
    transformCardContours(originalCard, cards, cardsContours);
    
    for (int i = 0; i < cards.size(); i++) {
        Mat card = cards[i];
        String closestCard;
        getClosestCard(card, cardset, closestCard);
		cout << "\nClosest card = " + closestCard << endl;
    }

    //waitKey(0);
    //destroyAllWindows();
    
    /*
    // Read camera
    
    // open default camera 
    VideoCapture cap(1);
    
    if(!cap.isOpened())
        cout << "Could not read camera" << endl;

    //namedWindow("Camera", 1);
    
    while(true)
    {
        Mat frame;
        // get a new frame from camera
        cap >> frame;
    
        imshow("Camera", frame);

        int key = waitKey(30);
        
        // Enter
        if(key == 10) {
            preProcessImage(frame);
            // Display transformation
            namedWindow("Transformed", WINDOW_AUTOSIZE );
            imshow("Transformed", originalCard);
            vector<vector<Point> > cardsContours;
            findContours(frame, numCards, cardsContours);
            vector<Mat> cards;
            transformCardContours(frame, cards, cardsContours);
        
            for (int i = 0; i < cards.size(); i++) {
                Mat card = cards[i];
                String closestCard;
                getClosestCard(card, cardset, closestCard);
                cout << "\nClosest card = " + closestCard << endl;
            }
        // Escape
        } else if (key == 27)
            break;

    }
    */
    /*
    // Create window

    namedWindow( "Imagem Original", WINDOW_AUTOSIZE );

	// Display image

	imshow( "Imagem Original", originalImage );

	// Print some image features

	cout << "ORIGINAL IMAGE" << endl;

    printImageFeatures( originalImage );

    
	// Destroy the windows

	destroyAllWindows();
    */

	return 0;
}
