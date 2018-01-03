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

    // Compute good threshold level
    double thresholdValue = originalImage.at<uchar>((int)(originalImage.rows / 2), 
            (int)(originalImage.cols / 2)) - 30;
    cout << thresholdValue << endl; 
    // Apply threshold
    int thresholdType = THRESH_BINARY;
    int maxValue = 255;
    threshold( originalImage, originalImage, 
            thresholdValue, maxValue, thresholdType);
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", originalImage ); 
    //waitKey(0);
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
        vector<vector<Point>> &cardsContours)
{
    Mat cannyOutput;
    vector<vector<Point> > contours;
    vector<Point> approxCurve;
    double size, epsilon;

    int mode = CV_RETR_TREE;
    int method = CV_CHAIN_APPROX_SIMPLE;

    Canny(image, cannyOutput, 120, 240);

    findContours(cannyOutput, contours, mode, method, Point(0, 0));
    
    for (int i = 0; i < contours.size(); i++) {
        size = contourArea(contours[i]);
        epsilon = 0.02 * arcLength(contours[i], true);
        approxPolyDP(contours[i], approxCurve, epsilon, true);

        if (size > 25000 && size < 120000 && approxCurve.size() == 4)
            cardsContours.push_back(contours[i]);
    }

    // Find the most common contours
    sort(cardsContours.begin(), cardsContours.end(), reverseCompareContourArea);
    //cardsContours.resize(numCards);
}

void transformCardContours(const Mat &image, vector<Mat> &cards,
        const vector<vector<Point>> &cardsContours)
{
    int centerX, centerY;
    double epsilon, width, heigth;
    vector<Point> contour;

    // Transform perspective card into a 500x500 image card
    for (int i = 0; i < cardsContours.size(); i++) {
        
        Mat card;

        contour = cardsContours[i];
        
        // Compute approximation accuracy
        vector<Point2f> approxCurve;
        double epsilon = 0.02 * arcLength(contour, true);
        approxPolyDP(contour, approxCurve, epsilon, true);
        
        // Get bounding rectangle and dimensions
        //Rect boundRect = boundingRect(contour);
        
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
        dstVertices[1] = Point2f(199, 0); 
        dstVertices[2] = Point2f(199, 299);        
        dstVertices[3] = Point2f(0, 299);        
        
        // Order vertices counter clockwise
        orderPointsCC(approxCurve, srcVertices);
        
        // Get the transformation matrix
        Mat transform = getPerspectiveTransform(srcVertices, dstVertices);
        // Perform the transformation
        warpPerspective(image, card, transform, Size(200, 300));
        
        if (card.channels() > 1)
	        cvtColor(card, card, COLOR_BGR2GRAY, 1);
        
        cards.push_back(card);
    }
}

bool processCorner(const Mat &card, Mat &rank, Mat &suit)
{
    Rect roi;
    
    // Get corner area 
    roi = Rect(Point(0, 0), Point(34, 84));
    Mat cardCorner = card(roi);
    resize(cardCorner, cardCorner, Point(0, 0), 4, 4);
    
    // Compute good threshold level
    double white = card.at<uchar>(15, (int)((33 * 4) / 2));
    double thresholdValue = white - 30 > 0 ? white - 30 : 1;
    
    // Apply threshold
    int thresholdType = THRESH_BINARY;
    threshold(cardCorner, cardCorner, thresholdValue, 255, thresholdType);
        
    // Get regions of interest
    roi = Rect(Point(0, 0), Point(135, 200));
    /*Mat drawing = cardCorner.clone();
    rectangle(drawing, roi, Scalar(128, 128, 128));
    namedWindow( "Imagem Original", WINDOW_AUTOSIZE );
    imshow("Imagem Original", drawing);
    */
    rank = cardCorner(roi);
    roi = Rect(Point(0, 180), Point(135, 336));
    suit = cardCorner(roi); 
    Rect box;
    vector<vector<Point>> contours;
    int mode = CV_RETR_TREE;
    int method = CV_CHAIN_APPROX_SIMPLE;
    
    try {
        // Rank processing
        findContours(rank, contours, mode, method, Point(0, 0));
        sort(contours.begin(), contours.end(), reverseCompareContourArea);

        // Use largest contour to resize
        box = boundingRect(contours[1]);
        roi = Rect(Point(box.x, box.y), Point(box.x + box.width, box.y + box.height));
        rank = rank(roi);
        resize(rank, rank, Point(70, 125));

        // Suit processing
        findContours(suit, contours, mode, method, Point(0, 0));
        sort(contours.begin(), contours.end(), reverseCompareContourArea);

        // Use largest contour to resize
        box = boundingRect(contours[1]);
        roi = Rect(Point(box.x, box.y), Point(box.x + box.width, box.y + box.height));
        
        /*
        Mat drawing = suit.clone();
        rectangle(drawing, roi, Scalar(128, 128, 128));
        namedWindow( "Imagem Original", WINDOW_AUTOSIZE );
        imshow("Imagem Original", drawing);
        for( int i = 0; i< contours.size(); i++ )
        {
            cout << i << endl;
            drawContours( drawing, contours, i, Scalar(128, 128, 128), 2, 8);
            imshow("Imagem Original", drawing);
            waitKey(0);
        }
        */

        
        suit = suit(roi);
        resize(suit, suit, Point(70, 100));
    } catch (Exception e) {
        return false;
    }

    imwrite("rank.jpg", rank);
    imwrite("suit.jpg", suit);
    
    return true;
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
            
            Mat image = imread(filename, IMREAD_GRAYSCALE);
            
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

void getClosestCard(Mat &cardRank, Mat &cardSuit, 
        map<string, Mat> &ranks, map<string, Mat> &suits, 
        string &cardName)
{
    int i = -1;
    int diff, tmpDiff;
    string rank, suit;
    map<string, Mat>::iterator it = ranks.begin();

    // Find closest rank
    while(it != ranks.end()) {
        
        /*
        // Display difference
        namedWindow( "Imagem Original"+i, WINDOW_AUTOSIZE );
	    imshow("Imagem Original"+i, abs(cardRank - it->second));
        */

        if (!++i){
            diff = countBinaryWhite(abs(cardRank - it->second));
            rank = it->first;
        } else {
            tmpDiff = countBinaryWhite(abs(cardRank - it->second));
            if (tmpDiff < diff) {
                diff = tmpDiff;
                rank = it->first;
            }
        }
        it++;
    }
   
    // Find closest suit
    i = -1;
    it = suits.begin();
    while(it != suits.end()) {
        /*
        // Display difference
        namedWindow( "Imagem Original"+i, WINDOW_AUTOSIZE );
	    imshow("Imagem Original"+i, abs(compare_card - card));
        */

        if (!++i){
            diff = countBinaryWhite(abs(cardSuit - it->second));
            suit = it->first;
        } else {
            tmpDiff = countBinaryWhite(abs(cardSuit - it->second));
            if (tmpDiff < diff) {
                diff = tmpDiff;
                suit = it->first;
            }
        }
        it++;
    }

    cardName = rank + " of " + suit;
}

int main( int argc, char** argv )
{

    // Get dataset
    int numCards = 1;
    map<string, Mat> rankSet; 
    map<string, Mat> suitSet; 
    getTrainingSet("./sets/rank_set/", rankSet);
    getTrainingSet("./sets/suit_set/", suitSet);
    
    /*if( argc != 2 )
    {
		cout << "The name of the image file is missing !!" << endl;

        return -1;
    }

    
    // Get dataset
    int numCards = 1;
    map<string, Mat> rankSet; 
    map<string, Mat> suitSet; 
    getTrainingSet("./sets/rank_set/", rankSet);
    getTrainingSet("./sets/suit_set/", suitSet);

    Mat originalCard;

	originalCard = imread( argv[1], IMREAD_UNCHANGED );

	if( originalCard.empty() )
	{
		// NOT SUCCESSFUL : the data attribute is empty
		cout << "Image file could not be open !!" << endl;
	    return -1;
	}
    
    preProcessImage(originalCard);
    
    // Display transformation
    namedWindow("Transformed", WINDOW_AUTOSIZE );
    imshow("Transformed", originalCard);

    vector<vector<Point> > cardsContours;
    findContours(originalCard, numCards, cardsContours);
    vector<Mat> cards;
    transformCardContours(originalCard, cards, cardsContours);
    
    Mat rank, suit;
    for (int i = 0; i < cards.size(); i++) {
        Mat card = cards[i];
        processCorner(card, rank, suit);
        string closestCard;
        getClosestCard(rank, suit, rankSet, suitSet, closestCard);
		cout << "\nClosest card = " + closestCard << endl;
    }

    waitKey(0);
    destroyAllWindows();
    */ 
    
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
        if(key == 13) {
            preProcessImage(frame);
            vector<vector<Point> > cardsContours;
            findContours(frame, numCards, cardsContours);
            vector<Mat> cards;
            transformCardContours(frame, cards, cardsContours);
            
            Mat rank, suit;
            int size = numCards < cards.size() ? numCards : cards.size();

            for (int i = 0; i < size; i++) {
                Mat card = cards[i];
                if (!processCorner(card, rank, suit))
                    continue;
                string closestCard;
                getClosestCard(rank, suit, rankSet, suitSet, closestCard);
                cout << "\nClosest card = " + closestCard << endl;
            }
        // Escape
        } else if (key == 27)
            break;

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
