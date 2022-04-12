#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include	"Supp.h"

using namespace tesseract;

int main()
{
    char* outText;
    Point TL, BR;
    TessBaseAPI* ocr = new TessBaseAPI();
    // Initialize tesseract-ocr with English, without specifying tessdata path
    if (ocr->Init("tessdata", "eng")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }

    // Open input image with leptonica library
    //Pix* image = pixRead("Inputs/Images/text1.png");
    Mat  image = imread("Inputs/Images/text2.jpg");
    cvtColor(image, image, COLOR_BGR2GRAY);
    ocr->SetImage((uchar*)image.data, image.size().width, image.size().height, image.channels(), image.step1());

    //Get the boxes: Sentences box
    Boxa* boxes = ocr->GetComponentImages(RIL_TEXTLINE, true, NULL, NULL);

    cvtColor(image, image, COLOR_GRAY2BGR);// in order to draw colored bounding box

    printf("Found %d textline image components.\n", boxes->n);
    for (int i = 0; i < boxes->n; i++) {
        BOX* box = boxaGetBox(boxes, i, L_CLONE);
        ocr->SetRectangle(box->x, box->y, box->w, box->h);
        char* ocrResult = ocr->GetUTF8Text();
        int conf = ocr->MeanTextConf();
        fprintf(stdout, "Box[%d]: x=%d, y=%d, w=%d, h=%d, confidence: %d, text: %s",
            i, box->x, box->y, box->w, box->h, conf, ocrResult);
        TL = Point(box->x, box->y);
        BR = Point(box->x + box->w, box->y + box->h);
        rectangle(image, TL, BR, Scalar(0, 0, 255));
        boxDestroy(&box);
    }

    // destroy the Tesseract object to free up memory:
    ocr->End();
    
    imshow("result", image);
    waitKey(0);
    destroyAllWindows();
    return 0;
}