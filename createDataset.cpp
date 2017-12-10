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


// AUXILIARY  FUNCTION

void printImageFeatures(const cv::Mat &image)
{
	std::cout << std::endl;

	std::cout << "Number of rows : " << image.size().height << std::endl;

	std::cout << "Number of columns : " << image.size().width << std::endl;

	std::cout << "Number of channels : " << image.channels() << std::endl;

	std::cout << "Number of bytes per pixel : " << image.elemSize() << std::endl;

	std::cout << std::endl;
}

void preProcessImage(const cv::Mat &originalImage)
{
    // Convert to a single-channel, intensity image
    if (originalImage.channels() > 1)
	    cv::cvtColor(originalImage, originalImage, cv::COLOR_BGR2GRAY, 1);

    // Apply Gaussian Filter to get only the countour of the symbols and
    // remove all the artifacts
    int n = 3;
    double sigmaX = 100.0;
    cv::GaussianBlur(orginalImage, originalImage, cv::Size(n, n), sigmaX);

    // Apply threshold
    int thresholdValue = 120;
    int thresholdType = cv::THRESH_BINARY;
    int maxValue = 255;
    cv::threshold( originalImage, originalImage, 
            thresholdValue, maxValue, thresholdType);
}

// From: https://stackoverflow.com/questions/13495207/opencv-c-sorting-contours-by-their-contourarea
bool reverseCompareContourArea(std::vector<cv::Point2f> c1, 
        std::vector<cv::Point2f> c2)
{
    return fabs(cv::contourArea(cv::Mat(c1))) 
        > fabs(cv::contourArea(cv::Mat(c2)));
}

void findContours(const cv::Mat &image, int numCards,
        const std::vector<std::vector<cv::Point2f> > &cardsContours)
{
    std::vector<std::vector<cv::Point2f> > contours;
    std::vector<cv::Vec4i> hierarchy;
    int mode = cv::CV_RETR_TREE;
    int method = cv::CV_CHAIN_APPROX_SIMPLE;

    findContours(image, contours, hierarchy, mode, method);

    // Find the most common contours
    std::sort(contours.begin(), contours.end(), reverseCompareContourArea);
    cardsContours(&contours[0], &contours[numCards]);
}

void transformCardContours(const cv::Mat &image, const std::vector<cv::Mat> &cards,
        const std::vector<std::vector<cv::Point> > &cardsContours)
{
    cv::Point2f *points, *perspectivePoints;
    cv::Mat card;

    // Transform perspective card into a 500x500 image card
    for (int i = 0; i < cardsContours.size(); i++) {
        points = &cardsContours[0][0];
        cv::getPerspectiveTransform(points, perspectivePoints);

        cv::warpPerspective(image, card, perspectivePoints, cv::Size(500, 500));
        cards.push_back(card);
    }
}

void learnCards(const std:vector<cv::Mat> &cards, 
        const std::vector<std::string> cardNames, 
        const std::map<std::string, cv::Mat> &cardDataset)
{
    if (cards.size() != cardNames.size())
        return;

    for (int i = 0; i < cards.size(); i++)
        cardDataset[cardNames.at(i)] = cards.at(i);
}

void getClosestCard(const cv::Mat &card, const std::map<std::string, cv::Mat> &cards, 
        const std::string &cardName)
{
    int i = -1;
    int diff = 0, tmpDiff = 0;
    std::map<std::string, cv::Mat>::iterator it = cards.begin();

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


// MAIN

int main( int argc, char** argv )
{
    if( argc != 2 )
    {
		std::cout << "The name of the image file is missing !!" << std::endl;

        return -1;
    }

	cv::Mat originalImage;

	originalImage = cv::imread( argv[1], cv::IMREAD_UNCHANGED );

	if( originalImage.empty() )
	{
		// NOT SUCCESSFUL : the data attribute is empty

		std::cout << "Image file could not be open !!" << std::endl;

	    return -1;
	}

	
	// Create window

    cv::namedWindow( "Imagem Original", cv::WINDOW_AUTOSIZE );

	// Display image

	cv::imshow( "Imagem Original", originalImage );

	// Print some image features

	std::cout << "ORIGINAL IMAGE" << std::endl;

    printImageFeatures( originalImage );

    
    
    // Waiting

    cv::waitKey( 0 );

	// Destroy the windows

	cv::destroyAllWindows();

	return 0;
}
