// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;

import android.graphics.Rect;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;

class Result {
    int classIndex;
    Float score;
    Rect rect;
    float[] mask;

    public Result(int cls, Float output, Rect rect, float[] mask) {
        this.classIndex = cls;
        this.score = output;
        this.rect = rect;
        this.mask =mask;
    }
};

public class PrePostProcessor {
    // for yolov5 model, no need to apply MEAN and STD
    public final static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
    public final static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};

    // model input image size
    public final static int INPUT_WIDTH = 640;
    public final static int INPUT_HEIGHT = 640;
    public final static int OUTPUT_COLUMN = 6; // left, top, right, bottom, score and label

    static String[] mClasses;

    static float[] resize_masks(float[] pixels,int w1, int h1, int w2,int h2)
    {
        float[] temp= new float[w2*h2];
        float x_ratio = w1/(float)w2 ;
        float y_ratio = h1/(float)h2 ;
        float px, py ;
        for (int i=0;i<h2;i++) {
            for (int j=0;j<w2;j++) {
                px = (float) Math.floor(j*x_ratio);
                py = (float) Math.floor(i*y_ratio) ;
                temp[(i*w2)+j] = pixels[(int)((py*w1)+px)] ;
            }
        }
        return temp;
    }


    static ArrayList<Result> outputsToPredictions(int countResult, float[] outputs, List<float[]> masks, float imgScaleX, float imgScaleY, float ivScaleX, float ivScaleY, float startX, float startY) {
        ArrayList<Result> results = new ArrayList<>();
        for (int i = 0; i< countResult; i++) {
            float left = outputs[i* OUTPUT_COLUMN];
            float top = outputs[i* OUTPUT_COLUMN +1];
            float right = outputs[i* OUTPUT_COLUMN +2];
            float bottom = outputs[i* OUTPUT_COLUMN +3];

            int w1=(int)(right-left);
            int h1=(int)(bottom-top);

            left = imgScaleX * left;
            top = imgScaleY * top;
            right = imgScaleX * right;
            bottom = imgScaleY * bottom;

            Rect rect = new Rect((int)(startX+ivScaleX*left), (int)(startY+top*ivScaleY), (int)(startX+ivScaleX*right), (int)(startY+ivScaleY*bottom));

            int w2=(int)(rect.right-rect.left);
            int h2=(int)(rect.bottom-rect.top);
            float[] mask=resize_masks(masks.get(i),w1,h1,w2,h2);
            //Log.d("D2Go",  "Box Modified cooridnates for index  : "+ i+ " "+ rect.left+" "+ rect.top+" "+ rect.right+" "+ rect.bottom);
            //Log.d("D2Go",  "Length of resized mask for index : "+ i+ " "+ mask.length);
            Result result = new Result((int)outputs[i* OUTPUT_COLUMN +5], outputs[i* OUTPUT_COLUMN +4], rect,mask);
            results.add(result);
        }
        return results;
    }
}
