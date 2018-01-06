/*
 * CARD RECOGNITION (MATCH CONTOURS VERSION)
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

void findContours(const Mat &image, int numCards,
        vector<vector<Point>> &cardsContours)
{
    Mat cannyOutput;
    int mode = CV_RETR_TREE;
    int method = CV_CHAIN_APPROX_SIMPLE;

    Canny(image, cannyOutput, 120, 240);

    findContours(cannyOutput, cardsContours, mode, method, Point(0, 0));
    Mat x = image.clone();
    Scalar color = Scalar(128, 128, 128);
}

void getTrainingSet(const string path, 
        map<string, vector<vector<Point>>> &cardDataset)
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
            vector<vector<Point> > cardContours;
            findContours(image, 1, cardContours);
            cardDataset[filename] = cardContours;
        }
    }
}

void getClosestCard(vector<vector<Point>> &card, map<string, vector<vector<Point>>> &cards, 
        String &cardName)
{
    int j = -1;
    int diff, tmpDiff;
    map<string, vector<vector<Point>>>::iterator it = cards.begin();

    while(it != cards.end()) {

        tmpDiff = 0;
        if (abs(card.size() - it->second.size()) < 200){

            int size = card.size() > it->second.size() ? it->second.size() : card.size();
            for (int i = 0; i < size; i++) {
                // Compute approximation accuracy
                double epsilon1 = 0.02 * arcLength(card[i], true);
                double epsilon2 = 0.02 * arcLength(it->second[i], true);
                vector<Point> approxCurve1, approxCurve2;
                approxPolyDP(card[i], approxCurve1, epsilon1, true);
                approxPolyDP(it->second[i], approxCurve2, epsilon2, true);
                tmpDiff += matchShapes(approxCurve1, approxCurve2, CV_CONTOURS_MATCH_I1, 0);
            }
            
            if (!++j){
                diff = abs(tmpDiff);
                cardName = it->first;
            } else {
                if (abs(tmpDiff) < diff) {
                    diff = tmpDiff;
                    cardName = it->first;
                }
            }
        }

        it++;
    }


}

int main( int argc, char** argv )
{
    
    // Get dataset
    int numCards = 1;
    map<string, vector<vector<Point>>> cardset; 
    getTrainingSet("./sets/training_set/", cardset);

    // Read camera
    VideoCapture cap(1);
    
    if(!cap.isOpened())
        cout << "Could not read camera" << endl;

    namedWindow("Camera", 1);
    
    while(true)
    {
        Mat frame;
        
        // Get a new frame from camera
        cap >> frame;
    
        imshow("Camera", frame);

        int key = waitKey(30);
        
        // Enter
        if(key == 13) {
            preProcessImage(frame);

            vector<vector<Point>> cardContours;
            findContours(frame, numCards, cardContours);
            Mat x = frame.clone();
            Scalar color = Scalar(128, 128, 128);

            String closestCard;
            getClosestCard(cardContours, cardset, closestCard);
		    cout << "\nClosest card = " + closestCard << endl;
        //Escape
        } else if (key == 27)
            break;

    }
    
	return 0;
}
