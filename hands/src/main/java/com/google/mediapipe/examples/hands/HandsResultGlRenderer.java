// Copyright 2021 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.mediapipe.examples.hands;

import static androidx.camera.core.CameraX.getContext;

import android.app.Application;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
//import android.media.MediaRouter2;
import android.opengl.GLES20;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
//import android.util.Log;
//import android.widget.TextView;

import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.solutioncore.ResultGlRenderer;
import com.google.mediapipe.solutions.hands.Hands;
import com.google.mediapipe.solutions.hands.HandsResult;

import org.tensorflow.lite.Interpreter;
//import org.tensorflow.lite.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
//import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.List;

/** A custom implementation of {@link ResultGlRenderer} to render {@link HandsResult}. */
public class HandsResultGlRenderer implements ResultGlRenderer<HandsResult> {

  private static final String TAG = "HandsResultGlRenderer";

  private static final float[] LEFT_HAND_CONNECTION_COLOR = new float[] {0.2f, 1f, 0.2f, 1f};
  private static final float[] RIGHT_HAND_CONNECTION_COLOR = new float[] {1f, 0.2f, 0.2f, 1f};
  private static final float CONNECTION_THICKNESS = 25.0f;
  private static final float[] LEFT_HAND_HOLLOW_CIRCLE_COLOR = new float[] {0.2f, 1f, 0.2f, 1f};
  private static final float[] RIGHT_HAND_HOLLOW_CIRCLE_COLOR = new float[] {1f, 0.2f, 0.2f, 1f};
  private static final float HOLLOW_CIRCLE_RADIUS = 0.01f;
  private static final float[] LEFT_HAND_LANDMARK_COLOR = new float[] {1f, 0.2f, 0.2f, 1f};
  private static final float[] RIGHT_HAND_LANDMARK_COLOR = new float[] {0.2f, 1f, 0.2f, 1f};
  private static final float LANDMARK_RADIUS = 0.008f;
  private static final int NUM_SEGMENTS = 120;
  private static final String MODEL_NAME = "model.tflite";
  private static final String VERTEX_SHADER =
          "uniform mat4 uProjectionMatrix;\n"
                  + "attribute vec4 vPosition;\n"
                  + "void main() {\n"
                  + "  gl_Position = uProjectionMatrix * vPosition;\n"
                  + "}";
  private static final String FRAGMENT_SHADER =
          "precision mediump float;\n"
                  + "uniform vec4 uColor;\n"
                  + "void main() {\n"
                  + "  gl_FragColor = uColor;\n"
                  + "}";
  private int program;
  private int positionHandle;
  private int projectionMatrixHandle;
  private int colorHandle;
  private char data;

  private int loadShader(int type, String shaderCode) {
    int shader = GLES20.glCreateShader(type);
    GLES20.glShaderSource(shader, shaderCode);
    GLES20.glCompileShader(shader);
    return shader;
  }

  @Override
  public void setupRendering() {
    program = GLES20.glCreateProgram();
    int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER);
    int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
    GLES20.glAttachShader(program, vertexShader);
    GLES20.glAttachShader(program, fragmentShader);
    GLES20.glLinkProgram(program);
    positionHandle = GLES20.glGetAttribLocation(program, "vPosition");
    projectionMatrixHandle = GLES20.glGetUniformLocation(program, "uProjectionMatrix");
    colorHandle = GLES20.glGetUniformLocation(program, "uColor");
  }

  @Override
  public void renderResult(HandsResult result, float[] projectionMatrix) {
    if (result == null) {
      return;
    }
    GLES20.glUseProgram(program);
    GLES20.glUniformMatrix4fv(projectionMatrixHandle, 1, false, projectionMatrix, 0);
    GLES20.glLineWidth(CONNECTION_THICKNESS);

    float[][] vertices = new float[21][3];

    int numHands = result.multiHandLandmarks().size();
    for (int i = 0; i < numHands; ++i) {
      boolean isLeftHand = result.multiHandedness().get(i).getLabel().equals("Left");
      drawConnections(
              result.multiHandLandmarks().get(i).getLandmarkList(),
              isLeftHand ? LEFT_HAND_CONNECTION_COLOR : RIGHT_HAND_CONNECTION_COLOR);
//              System.out.print(result.multiHandLandmarks().get(i).getLandmarkList());
      for (NormalizedLandmark landmark : result.multiHandLandmarks().get(i).getLandmarkList()) {
        // Draws the landmark.
        drawCircle(
                landmark.getX(),
                landmark.getY(),
                isLeftHand ? LEFT_HAND_LANDMARK_COLOR : RIGHT_HAND_LANDMARK_COLOR);
        // Draws a hollow circle around the landmark.
        drawHollowCircle(
                landmark.getX(),
                landmark.getY(),
                isLeftHand ? LEFT_HAND_HOLLOW_CIRCLE_COLOR : RIGHT_HAND_HOLLOW_CIRCLE_COLOR);
      }
      int j = 0;
      for (NormalizedLandmark pointLandmark : result.multiHandLandmarks().get(i).getLandmarkList()) {
        vertices[j][0] = pointLandmark.getX();
        vertices[j][1] = pointLandmark.getY();
        vertices[j][2] = pointLandmark.getZ();
        j += 1;
      }
      makeAngle(vertices);
    }
  }

  /**
   * Deletes the shader program.
   *
   * <p>This is only necessary if one wants to release the program while keeping the context around.
   */
  public void release() {
    GLES20.glDeleteProgram(program);
  }

  private void drawConnections(List<NormalizedLandmark> handLandmarkList, float[] colorArray) {
    GLES20.glUniform4fv(colorHandle, 1, colorArray, 0);
    for (Hands.Connection c : Hands.HAND_CONNECTIONS) {
      NormalizedLandmark start = handLandmarkList.get(c.start());
      NormalizedLandmark end = handLandmarkList.get(c.end());
      float[] vertex = {start.getX(), start.getY(), end.getX(), end.getY()};
      FloatBuffer vertexBuffer =
              ByteBuffer.allocateDirect(vertex.length * 4)
                      .order(ByteOrder.nativeOrder())
                      .asFloatBuffer()
                      .put(vertex);
      vertexBuffer.position(0);
      GLES20.glEnableVertexAttribArray(positionHandle);
      GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer);
      GLES20.glDrawArrays(GLES20.GL_LINES, 0, 2);
    }
  }

  private void drawHollowCircle(float x, float y, float[] colorArray) {
    GLES20.glUniform4fv(colorHandle, 1, colorArray, 0);
    int vertexCount = NUM_SEGMENTS + 1;
    float[] vertices = new float[vertexCount * 3];
    for (int i = 0; i < vertexCount; i++) {
      float angle = 2.0f * i * (float) Math.PI / NUM_SEGMENTS;
      int currentIndex = 3 * i;
      vertices[currentIndex] = x + (float) (HOLLOW_CIRCLE_RADIUS * Math.cos(angle));
      vertices[currentIndex + 1] = y + (float) (HOLLOW_CIRCLE_RADIUS * Math.sin(angle));
      vertices[currentIndex + 2] = 0;
    }
    FloatBuffer vertexBuffer =
            ByteBuffer.allocateDirect(vertices.length * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer()
                    .put(vertices);
    vertexBuffer.position(0);
    GLES20.glEnableVertexAttribArray(positionHandle);
    GLES20.glVertexAttribPointer(positionHandle, 3, GLES20.GL_FLOAT, false, 0, vertexBuffer);
    GLES20.glDrawArrays(GLES20.GL_LINE_STRIP, 0, vertexCount);
  } //drawHollowCircle

  private void drawCircle(float x, float y, float[] colorArray) {
    GLES20.glUniform4fv(colorHandle, 1, colorArray, 0);
    int vertexCount = NUM_SEGMENTS + 2;
    float[] vertices = new float[vertexCount * 3];
    vertices[0] = x;
    vertices[1] = y;
    vertices[2] = 0;
    for (int i = 1; i < vertexCount; i++) {
      float angle = 2.0f * i * (float) Math.PI / NUM_SEGMENTS;
      int currentIndex = 3 * i;
      vertices[currentIndex] = x + (float) (LANDMARK_RADIUS * Math.cos(angle));
      vertices[currentIndex + 1] = y + (float) (LANDMARK_RADIUS * Math.sin(angle));
      vertices[currentIndex + 2] = 0;
    }
    FloatBuffer vertexBuffer =
            ByteBuffer.allocateDirect(vertices.length * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer()
                    .put(vertices);
    vertexBuffer.position(0);
    GLES20.glEnableVertexAttribArray(positionHandle);
    GLES20.glVertexAttribPointer(positionHandle, 3, GLES20.GL_FLOAT, false, 0, vertexBuffer);
    GLES20.glDrawArrays(GLES20.GL_TRIANGLE_FAN, 0, vertexCount);
  } //drawCircle


  private void makeAngle(float[][] vertices) { //각도구하는 함수
    float[][] joint = new float[21][3]; // 21개 점의 x,y,z의 좌표
    float[][] v1 = new float[20][3]; //v1_n
    float[][] v2 = new float[20][3]; //v2_n
    float[][] v = new float[20][3]; //v2-v1 길이
    float[][] v1_new = new float[15][3];
    float[][] v2_new = new float[15][3];
    int[] compareV1 = {0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17}; //15개  노말라이즈해서 크기 1의 벡터가 나오게 됨.
    int[] compareV2 = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19}; //15개
    int[] v1_n = {0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19}; //20개
    int[] v2_n = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}; //20개

    for (int i = 0; i < joint.length; i++) {
      for (int j = 0; j < joint[i].length; j++) {
        joint[i][j] = vertices[i][j];
      }
    }
    for (int i = 0; i < v.length; i++) {
      for (int j = 0; j < v[i].length; j++) {
        v1[i][j] = vertices[v1_n[i]][j]; // joint[v1_n[19]]까지
        v2[i][j] = vertices[v2_n[i]][j];
      }
      //Log.i(TAG, String.format("***v1*** %f %f %f",v1[i][0], v1[i][1], v1[i][2]));
      //Log.i(TAG, String.format("***v2*** %f %f %f",v2[i][0], v2[i][1], v2[i][2]));
    }
    for (int i = 0; i < v.length; i++) {
      for (int j = 0; j < v[i].length; j++) {
        v[i][j] = v2[i][j] - v1[i][j];
      }
//      Log.i(TAG, String.format("***v*** %f %f %f",v[i][0], v[i][1], v[i][2]));
    }

    float[] powCal = new float[20];
    float[] powSum = new float[20];
    //v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
    for (int i = 0; i < v.length; i++) {
      for (int j = 0; j < v[i].length; j++){
        powCal[i] += Math.pow(v[i][j], 2);
      }
      powSum[i] = (float) Math.sqrt(powCal[i]);
    }
    for (int i = 0; i < v.length; i++){
      for (int k = 0; k < v[i].length; k++) {
        v[i][k] = powSum[i]/powCal[i]*v[i][k];
      }
    }


    for (int i = 0; i < v1_new.length; i++) {
      for (int j = 0; j < v1_new[i].length; j++) {
        v1_new[i][j] = v[compareV1[i]][j];
        v2_new[i][j] = v[compareV2[i]][j];
      }
    }

    //palmAngle
    float[] nine = {joint[9][0], joint[9][1]};
    float[] zero = {joint[0][0], joint[0][1]};
    float[] zeroaxis0 = {joint[0][0]+10, joint[0][1]};

    float radians = 0;
    radians = (float) (Math.atan2(zeroaxis0[1]-zero[1], zeroaxis0[0]-zero[0]) - Math.atan2(nine[1]-zero[1], nine[0]-zero[0]));
    float p_angle = 0;
    p_angle = (float) Math.abs(radians*180.0/Math.PI);


    float[] sum = new float[15];
    float[] angle = new float[15];
    float[][] result = new float[1][16]; // 1 x 16

    for (int i = 0; i < compareV1.length; i++) { //내적
      for (int j = 0; j < joint[i].length; j++) {
        sum[i] += v1_new[i][j] * v2_new[i][j];
      }
      angle[i] = (float) Math.acos(sum[i]);
      result[0][i] = (float) Math.toDegrees(angle[i]);
    }
    result[0][15] = (float) p_angle; // 16번째 자리에 손바닥 각도 넣음
    //Log.i(TAG, String.format("***result*** %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f",result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11], result[12], result[13], result[14], result[15]));

    Interpreter tflite = getTfliteInterpreter(MODEL_NAME);

    float[][] output = new float[1][31]; // 출력 결과 31개

    tflite.run(result, output); // input:result 16개(관절15+손바닥1) output 31개
    //Log.i(TAG, String.format("***index*** %d ",output[0][0]));

    float max = output[0][0];
    int maxIndex = 0;

    char[] gesture = {'ㄱ',  'ㄴ',  'ㄷ',  'ㄹ',  'ㅁ',  'ㅂ',  'ㅅ',  'ㅇ',
            'ㅈ',  'ㅊ',  'ㅋ',  'ㅌ',  'ㅍ',  'ㅎ',  'ㅏ',
            'ㅑ',  'ㅓ',  'ㅕ',  'ㅗ',  'ㅛ',  'ㅜ',  'ㅠ',
            'ㅡ',  'ㅣ',  'ㅐ',  'ㅒ',  'ㅔ',  'ㅖ',  'ㅢ',  'ㅚ',  'ㅟ'};

    for (int i=0 ; i < output[0].length; i++) {

      if (output[0][i] > max) {
        max = output[0][i];
        maxIndex = i;
      }
      data = gesture[maxIndex]; // data 이년을 전달해야됨

      //Toast.makeText(getContext(), data, Toast.LENGTH_SHORT).show();
      //Log.i(TAG, String.format("***index*** %c ", gesture[maxIndex]));
    }

  } //makeAngle

  private Interpreter getTfliteInterpreter(String modelPath){
    try {
      return new Interpreter((loadModelFile(modelPath))); //loadmodelfile 함수에 예외가 포함되어 있기 때문에 반드시 try/catch
    }
    catch (Exception e){
      e.printStackTrace();
    }
    return null;
  }

  private ByteBuffer loadModelFile(String modelPath) throws IOException { // tflite 파일 읽어오기
    AssetManager am = getContext().getAssets();
    AssetFileDescriptor afd = null;
    try {
      afd = am.openFd(modelPath);
    } catch (IOException e) {
      e.printStackTrace();
    }
    assert afd != null;
    FileInputStream fis = new FileInputStream(afd.getFileDescriptor());
    FileChannel fc = fis.getChannel();
    long startOffset = afd.getStartOffset();
    long declaredLength = afd.getDeclaredLength();
    return fc.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }
}

