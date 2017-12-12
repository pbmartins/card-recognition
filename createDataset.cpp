/*
 * createDataset.cpp
 *
 * DATASET GENERATOR
 *
 * Diogo Ferreira - Pedro Martins - 2017
 */


#include <iostream>

#include <math.h>

#include <stdlib.h>

#include "opencv2/core/core.hpp"

#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;


void printImageFeatures(const Mat &image)
{
	cout << endl;

	cout << "Number of rows : " << image.size().height << endl;

	cout << "Number of columns : " << image.size().width << endl;

	cout << "Number of channels : " << image.channels() << endl;

	cout << "Number of bytes per pixel : " << image.elemSize() << endl;

	cout << endl;
}

void preProcessImage(const Mat &originalImage)
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
    int thresholdValue = 120;
    int thresholdType = THRESH_BINARY;
    int maxValue = 255;
    threshold( originalImage, originalImage, 
            thresholdValue, maxValue, thresholdType);
}

// From: https://stackoverflow.com/questions/13495207/opencv-c-sorting-contours-by-their-contourarea
bool reverseCompareContourArea(vector<Point2f> c1, 
        vector<Point2f> c2)
{
    return fabs(contourArea(Mat(c1))) 
        > fabs(contourArea(Mat(c2)));
}

void findContours(const Mat &image, int numCards,
        const vector<vector<Point2f> > &cardsContours)
{
    vector<vector<Point2f> > contours;
    vector<Vec4i> hierarchy;
    int mode = RETR_TREE;
    int method = CHAIN_APPROX_SIMPLE;

    findContours(image, contours, hierarchy, mode, method);

    // Find the most common contours
    sort(contours.begin(), contours.end(), reverseCompareContourArea);
    //cardsContours = contours;
    //cardsContours(&contours[0], &contours[numCards]);
}

void transformCardContours(const Mat &image, const vector<Mat> &cards,
        const vector<vector<Point> > &cardsContours)
{
    Point2f *points, *perspectivePoints;
    Mat card;

    // Transform perspective card into a 500x500 image card
    for (int i = 0; i < cardsContours.size(); i++) {
        points = cardsContours[i];
        getPerspectiveTransform(points, perspectivePoints);

        warpPerspective(image, card, perspectivePoints, Size(500, 500));
        cards.push_back(card);
    }
}

void learnCards(const std:vector<Mat> &cards, 
        const vector<string> cardNames, 
        const map<string, Mat> &cardDataset)
{
    if (cards.size() != cardNames.size())
        return;

    for (int i = 0; i < cards.size(); i++)
        cardDataset[cardNames.at(i)] = cards.at(i);
}

void getClosestCard(const Mat &card, const map<string, Mat> &cards, 
        const string &cardName)
{
    int i = -1;
    int diff = 0, tmpDiff = 0;
    map<string, Mat>::iterator it = cards.begin();

    while(it != cards.end()) {
        if (!++i)
            diff = abs(card - it->second);
        else {
            tmpDiff = abs(card - it->second);
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
    /*
    if( argc != 2 )
    {
		cout << "The name of the image file is missing !!" << endl;

        return -1;
    }

    Mat originalImage;

	originalImage = imread( argv[1], IMREAD_UNCHANGED );

	if( originalImage.empty() )
	{
		// NOT SUCCESSFUL : the data attribute is empty

		cout << "Image file could not be open !!" << endl;

	    return -1;
	}
    */

    /* Read camera */
    
    // open default camera 
    VideoCapture cap(0);
    
    if(!cap.isOpened())
        cout << "Could not read camera" << endl;

    namedWindow("Camera", 1);
    
    while(true)
    {
        Mat frame;
        // get a new frame from camera
        cap >> frame;
        imshow("Camera", frame);
        if(waitKey(30) >= 0) break;
    }
    
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
