/*
 * CARD RECOGNITION (CORNER DETECTION)
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
    int n = 5;
    double sigmaX = 100.0;
    GaussianBlur(originalImage, originalImage, Size(n, n), sigmaX);

    // Apply OTSU threshold
    int thresholdType = THRESH_BINARY | THRESH_OTSU;
    threshold(originalImage, originalImage, 0, 255, thresholdType);
    
    // Dilate image to improve contours
    int size = 2;
    Mat element = getStructuringElement(MORPH_RECT, Size(size, size));
    dilate(originalImage, originalImage, element);
}

// From: https://stackoverflow.com/questions/13495207/opencv-c-sorting-contours-by-their-contourarea
bool reverseCompareContourArea(vector<Point> c1, 
        vector<Point> c2)
{
    return contourArea(c1, false) > contourArea(c2, false);
}

// Adapted from: https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
void orderPoints(const vector<Point2f> &points, Point2f* orderedPoints,
        double width, double height) 
{

    // Correct output order 
    //  3--2
    //  |  |
    //  0--1
    
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
    
    if (width <= 0.8 * height) {
        // Vertically oriented
        orderedPoints[0] = points[sMin];
        orderedPoints[1] = points[dMax];
        orderedPoints[2] = points[sMax];
        orderedPoints[3] = points[dMin];
    } else if (width >= 1.2 * height) {
        // Horizontally oriented
        orderedPoints[0] = points[dMin];
        orderedPoints[1] = points[sMin];
        orderedPoints[2] = points[dMax];
        orderedPoints[3] = points[sMax];
    } else {
        // Titled to left
        double dif = points[1].y - points[3].y;
        if (points[1].y <= points[3].y || abs(dif) < 40 ) {
            orderedPoints[0] = points[1];
            orderedPoints[1] = points[0];
            orderedPoints[2] = points[3];
            orderedPoints[3] = points[2];
        // Tilted to right
        } else {
            orderedPoints[0] = points[0];
            orderedPoints[1] = points[3];
            orderedPoints[2] = points[2];
            orderedPoints[3] = points[1];
        }
    }
}

void findCardsContours(const Mat &image, vector<vector<Point>> &cardsContours)
{
    Mat cannyOutput;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Point> approxCurve;
    double size, epsilon;

    int mode = CV_RETR_TREE;
    int method = CV_CHAIN_APPROX_SIMPLE;
    
    // Apply canny to improve contours
    Canny(image, cannyOutput, 120, 240);

    // Find card contours
    findContours(cannyOutput, contours, hierarchy, mode, method, Point(0, 0));

    // Post-process contours to find which corresponds to a card contour
    for (int i = 0; i < contours.size(); i++) {
        size = contourArea(contours[i]);
        epsilon = 0.02 * arcLength(contours[i], true);
        approxPolyDP(contours[i], approxCurve, epsilon, true);

        // A contour is a card contour if it has a size between 25*10^3 and 25*10^4
        // if it hasn't a parent and its approximation has 4 points (closed contour)
        if (size > 25000 && size < 250000 && hierarchy[i][3] == -1
               && approxCurve.size() == 4)
            cardsContours.push_back(contours[i]);
    }

    // Find the most common contours
    sort(cardsContours.begin(), cardsContours.end(), reverseCompareContourArea);
}

void transformCardContours(const Mat &image, vector<Mat> &cards, 
        vector<Point> &centers, const vector<vector<Point>> &cardsContours)
{
    double epsilon;
    Rect boundRect;
    Point center;
    vector<Point2f> approxCurve;
    vector<Point> contour;
    Mat card;

    // Transform perspective card into a 200x300 image card
    for (int i = 0; i < cardsContours.size(); i++) {

        contour = cardsContours[i];
        
        // Compute approximation accuracy
        epsilon = 0.02 * arcLength(contour, true);
        approxPolyDP(contour, approxCurve, epsilon, true);
        
        // Get bounding rectangle and it's center 
        boundRect = boundingRect(contour);
        center = (boundRect.br() + boundRect.tl()) / 2;

        /*
        // Debug rect
        Mat drawing = image.clone();
        Scalar color = Scalar(64, 64, 64);
        // Draw contours
        for (int j=0; j < cardsContours.size(); j++)
            drawContours(drawing, cardsContours, j, color, 5, 8, vector<Vec4i>(), 0, Point());
        // Draw bounding rectangle
        rectangle(drawing, boundRect, color);
         
        namedWindow("Rect Contours", CV_WINDOW_AUTOSIZE);
        imshow("Rect Contours", drawing); 
        */
            
        Point2f srcVertices[4];
        Point2f dstVertices[4];
        
        // Vertices of the corrected card position box 
        dstVertices[0] = Point2f(0, 0);
        dstVertices[1] = Point2f(199, 0); 
        dstVertices[2] = Point2f(199, 299);        
        dstVertices[3] = Point2f(0, 299);        
        
        // Order vertices counter clockwise, based on it's initial position
        orderPoints(approxCurve, srcVertices, boundRect.width, boundRect.height);
        
        // Get the transformation matrix
        Mat transform = getPerspectiveTransform(srcVertices, dstVertices);

        // Perform the transformation
        warpPerspective(image, card, transform, Size(200, 300));

        // Convert card color
        if (card.channels() > 1)
	        cvtColor(card, card, COLOR_BGR2GRAY, 1);
        
        // Save card color and center
        centers.push_back(center);
        cards.push_back(card);
    }
}

bool processCorner(const Mat &card, Mat &rank, Mat &suit)
{
    Rect roi;
    
    // Get corner area and zoom it 
    roi = Rect(Point(0, 0), Point(33, 90));
    Mat cardCorner = card(roi);
    resize(cardCorner, cardCorner, Point(0, 0), 4, 4);
    
    // Apply OTSU threshold
    int thresholdType = THRESH_BINARY | THRESH_OTSU;
    threshold(cardCorner, cardCorner, 0, 255, thresholdType);
    
    // Opening background to improve contours 
    int size = 7;
    Mat element = getStructuringElement(MORPH_RECT, Size(size, size));
    erode(cardCorner, cardCorner, element);
    element = getStructuringElement(MORPH_RECT, Size(size, size));
    dilate(cardCorner, cardCorner, element);
    
    /*  
    // Display corner
    namedWindow("Threshold corner", WINDOW_AUTOSIZE);
    imshow("Threshold corner", cardCorner);
    */

    // Create frame to add to rank and suit
    Mat cols(cardCorner.rows, 10, CV_8U, Scalar(255, 255, 255));
    if (cols.channels() > 1)
        cvtColor(cols, cols, COLOR_BGR2GRAY);
    hconcat(cardCorner, cols, cardCorner);

    Mat rows(20, suit.cols, CV_8U, Scalar(255, 255, 255));
    if (rows.channels() > 1)
        cvtColor(rows, rows, COLOR_BGR2GRAY);
    vconcat(rows, suit, suit);
   
    // Get regions of interest (rank and suit) by dividing the corner
    roi = Rect(Point(0, 0), Point(cardCorner.cols, 220));    
    rank = cardCorner(roi);
    roi = Rect(Point(0, 170), Point(cardCorner.cols, cardCorner.rows));
    suit = cardCorner(roi);
    
    Rect box;
    vector<vector<Point>> contours;
    int mode = CV_RETR_TREE;
    int method = CV_CHAIN_APPROX_SIMPLE;
    
    try {
        // Rank processing
        findContours(rank, contours, mode, method, Point(0, 0));
        // Bigger contour is the frame contour 
        if (contours.size() < 2)
            return false;
        
        sort(contours.begin(), contours.end(), reverseCompareContourArea);

        // Use largest contour (except frame contour) to resize
        box = boundingRect(contours[1]);
        roi = Rect(Point(box.x, box.y), Point(box.x + box.width, box.y + box.height));
        rank = rank(roi);
        resize(rank, rank, Point(70, 125));
        
        // Suit processing
        findContours(suit, contours, mode, method, Point(0, 0));
        // Bigger contour is the frame contour 
        if (contours.size() < 2)
            return false;
        
        sort(contours.begin(), contours.end(), reverseCompareContourArea);
        
        // Use largest contour (except frame contour) to resize
        box = boundingRect(contours[1]);
        roi = Rect(Point(box.x, box.y), Point(box.x + box.width, box.y + box.height));
        suit = suit(roi);
        resize(suit, suit, Point(70, 100));

    } catch (Exception e) {
        return false;
    }
    
    /*
    // Display rank and suit
    namedWindow("Rank", WINDOW_AUTOSIZE);
    imshow("Rank", rank);

    namedWindow("Suit", WINDOW_AUTOSIZE );
    imshow("Suit", suit);
    */

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
            
            cardDataset[filename] = imread(filename, IMREAD_GRAYSCALE);
        }
    }
}

int countBinaryWhite(Mat card) 
{
    // Count number of binary pixels in the image
    int count = 0;
    for (int i = 0; i < card.rows; i++) {
        for (int j = 0; j < card.cols; j++) {
            if (card.at<uchar>(i, j) == 255)
                count++;
        }
    }
    return count;
}

void getCardNameFromPath(string &path)
{
    // Remove directory
    size_t lastSlash = path.find_last_of('/');
    if (string::npos != lastSlash)
        path.erase(0, lastSlash + 1);

    // Remove extension
    size_t period = path.rfind('.');
    if (string::npos != period)
        path.erase(period);
}

bool getClosestCard(Mat &cardRank, Mat &cardSuit, 
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
        namedWindow("Rank difference", WINDOW_AUTOSIZE);
	    imshow("Rank difference", abs(cardRank - it->second));
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
    
    // Rank difference too big
    if (diff > 3000)
        return false;

    // Find closest suit
    i = -1;
    it = suits.begin();
    while(it != suits.end()) {
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
    
    // Suit difference too big
    if (diff > 700)
        return false;
  
    // Get card name from the file path 
    getCardNameFromPath(rank);
    getCardNameFromPath(suit);
    
    cardName = rank + " " + suit;
    return true;
}

void identify(Mat &image, Point &center, string cardName)
{   
    // Divide card name into rank and suit
    vector<string> names;
    stringstream ss(cardName);
    string token;
    while (getline(ss, token, ' ')) {
        names.push_back(token);
    }

    // Draw text with cardname in image (with black outline)
    putText(image, names[0] + " of ", Point(center.x - 60, center.y - 10),
             FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 5); 
    putText(image, names[0] + " of ", Point(center.x - 60, center.y - 10),
             FONT_HERSHEY_DUPLEX, 1, Scalar(128, 128, 128), 2);

    putText(image, names[1], Point(center.x - 60, center.y + 25),
             FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 5);
    putText(image, names[1], Point(center.x - 60, center.y + 25),
             FONT_HERSHEY_DUPLEX, 1, Scalar(128, 128, 128), 2);
}

int main( int argc, char** argv )
{

    // Create cardset
    map<string, Mat> rankSet; 
    map<string, Mat> suitSet; 
    getTrainingSet("./sets/rank_set/", rankSet);
    getTrainingSet("./sets/suit_set/", suitSet);
    
    // Read camera
    VideoCapture cap(0);
    if(!cap.isOpened())
        cout << "Could not read camera" << endl;

    namedWindow("Card Detector", WINDOW_AUTOSIZE);
   
    while(true)
    {
        Mat frame, originalFrame;
         
        /// Get a new frame from camera
        cap >> frame;
        originalFrame = frame.clone();

        // Pre-process image
        preProcessImage(frame);
        
        /* 
        // Display pre-processed image
        namedWindow("Pre-processed", WINDOW_AUTOSIZE);
        imshow("Pre-processed", frame);
        */

        // Find frame contours
        vector<vector<Point>> cardsContours;
        findCardsContours(frame, cardsContours);
        
        // Get cards in the image
        vector<Point> centers;
        vector<Mat> cards;
        transformCardContours(originalFrame, cards, centers, cardsContours);
        
        Point center;
        Mat card, rank, suit;
        for (int i = 0; i < cards.size(); i++) {
            card = cards[i];
            center = centers[i];

            /*
            // Display the card found
            namedWindow("Card"+i, WINDOW_AUTOSIZE);
            imshow("Card"+i, card);
            */

            // Get rank and suit through card corner
            if (!processCorner(card, rank, suit))
                continue;

            // Find the closest card
            string closestCard;
            if (!getClosestCard(rank, suit, rankSet, suitSet, closestCard))
                continue;
            
            // Draw in the frame the name of the card
            identify(originalFrame, center, closestCard);
        }

        imshow("Card Detector", originalFrame);
        
        int key = waitKey(1);
        // Escape
        if (key == 27)
            break;
    }
	
    return 0;
}
