/*
 * createDataset.cpp
 *
 * DATASET GENERATOR
 *
 * Diogo Ferreira - Pedro Martins - 2017
 */


#include <iostream>

#include <math.h>

#include <map>

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
        vector<vector<Point> > &cardsContours)
{
    vector<vector<Point> > contours;
    //vector<Vec4i> hierarchy;
    Mat cannyOutput;
    int mode = CV_RETR_TREE;
    int method = CV_CHAIN_APPROX_SIMPLE;

    printf("Finding contours");
    Canny(image, cannyOutput, 120, 240);
    printf("Applied canny");

    //findContours(cannyOutput, contours, hierarchy, mode, method);
    findContours(cannyOutput, cardsContours, mode, method, Point(0, 0));

    printf("Countours found");
    // Find the most common contours
    //sort(contours.begin(), contours.end(), reverseCompareContourArea);
    //cardsContours = contours;
    //cardsContours.resize(numCards);
    printf("\nCards resized");
}

void transformCardContours(const Mat &image, vector<Mat> &cards,
        const vector<vector<Point> > &cardsContours)
{
    vector<Point> points, perspectivePoints;
    Mat card;

    // Transform perspective card into a 500x500 image card
    for (int i = 0; i < cardsContours.size(); i++) {
        points = cardsContours[i];
        printf("finding perspective for %d", i);

        // TODO: function not correctly called
        getPerspectiveTransform(points, perspectivePoints);
        printf("Found perspective for %d", i);


        warpPerspective(image, card, perspectivePoints, Size(500, 500));
        printf("Found warp perspective for %d", i);

        cards.push_back(card);
    }
}

void learnCards(const vector<Mat> &cards, 
        const vector<string> cardNames, 
        map<string, Mat> &cardDataset)
{
    if (cards.size() != cardNames.size())
        return;

    for (int i = 0; i < cards.size(); i++)
        cardDataset[cardNames.at(i)] = cards.at(i);
}

int countBinaryWhite(Mat card) {
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
        string &cardName)
{
    int i = -1;
    int diff, tmpDiff;
    map<string, Mat>::iterator it = cards.begin();

    while(it != cards.end()) {
        if (!++i)
            diff = countBinaryWhite(abs(card - it->second));
        else {
            tmpDiff = countBinaryWhite(abs(card - it->second));
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


    Mat originalImage;

	originalImage = imread( argv[1], IMREAD_UNCHANGED );

	if( originalImage.empty() )
	{
		// NOT SUCCESSFUL : the data attribute is empty

		cout << "Image file could not be open !!" << endl;

	    return -1;
	}

    preProcessImage(originalImage);
    vector<vector<Point> > cardsContours;
    findContours(originalImage, 1, cardsContours);
    vector<Mat> cards;
    transformCardContours(originalImage, cards, cardsContours);
    
    Mat img = cards[0];

    namedWindow( "Imagem Original", WINDOW_AUTOSIZE );
	imshow( "Imagem Original", img );
    waitKey(0);
    destroyAllWindows();
    

    /* Read camera */
    
    // open default camera 
    /*VideoCapture cap(0);
    
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
    }*/
    
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
