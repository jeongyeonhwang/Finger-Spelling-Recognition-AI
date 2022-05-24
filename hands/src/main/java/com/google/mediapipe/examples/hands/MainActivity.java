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

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Bundle;
import android.provider.MediaStore;
import androidx.appcompat.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.exifinterface.media.ExifInterface;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.google.mediapipe.formats.proto.LandmarkProto;
import com.google.mediapipe.formats.proto.LandmarkProto.Landmark;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.solutioncore.CameraInput;
import com.google.mediapipe.solutioncore.SolutionGlSurfaceView;
import com.google.mediapipe.solutioncore.VideoInput;
import com.google.mediapipe.solutions.hands.HandLandmark;
import com.google.mediapipe.solutions.hands.Hands;
import com.google.mediapipe.solutions.hands.HandsOptions;
import com.google.mediapipe.solutions.hands.HandsResult;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;

/** Main activity of MediaPipe Hands app. */
public class MainActivity extends AppCompatActivity {
  private static final String TAG = "MainActivity";
  private static final String MODEL_NAME = "finger_model.tflite";
  public static String data;
  public String[] gesture = {"ㄱ",  "ㄴ",  "ㄷ",  "ㄹ",  "ㅁ",  "ㅂ",  "ㅅ",  "ㅇ",
          "ㅈ",  "ㅊ",  "ㅋ",  "ㅌ",  "ㅍ",  "ㅎ",  "ㅏ",
          "ㅑ",  "ㅓ",  "ㅕ",  "ㅗ",  "ㅛ",  "ㅜ",  "ㅠ",
          "ㅡ",  "ㅣ",  "ㅐ",  "ㅒ",  "ㅔ",  "ㅖ",  "ㅢ",  "ㅚ",  "ㅟ"};
  private FirebaseDatabase database = FirebaseDatabase.getInstance();
  private DatabaseReference myRef = database.getReference();

  private Hands hands;
  // Run the pipeline and the model inference on GPU or CPU.
  private static final boolean RUN_ON_GPU = true;
  //Classifier cls; /***********************/

  private enum InputSource {
    UNKNOWN,
    IMAGE,
    CAMERA,
  }
  private InputSource inputSource = InputSource.UNKNOWN;

  // Image demo UI and image loader components.
  private ActivityResultLauncher<Intent> imageGetter;
  private HandsResultImageView imageView;
  // Video demo UI and video loader components.
  private VideoInput videoInput;
  private ActivityResultLauncher<Intent> videoGetter;
  // Live camera demo UI and camera components.
  private CameraInput cameraInput;

  private SolutionGlSurfaceView<HandsResult> glSurfaceView;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    setupStaticImageDemoUiComponents();
    //setupVideoDemoUiComponents();
    setupLiveDemoUiComponents();
  }

  @Override
  protected void onResume() {
    super.onResume();
    if (inputSource == InputSource.CAMERA) {
      // Restarts the camera and the opengl surface rendering.
      cameraInput = new CameraInput(this);
      cameraInput.setNewFrameListener(textureFrame -> hands.send(textureFrame));
      glSurfaceView.post(this::startCamera);
      glSurfaceView.setVisibility(View.VISIBLE);
    }
  }

  @Override
  protected void onPause() {
    super.onPause();
    if (inputSource == InputSource.CAMERA) {
      glSurfaceView.setVisibility(View.GONE);
      cameraInput.close();
    }
  }

  private Bitmap downscaleBitmap(Bitmap originalBitmap) {
    double aspectRatio = (double) originalBitmap.getWidth() / originalBitmap.getHeight();
    int width = imageView.getWidth();
    int height = imageView.getHeight();
    if (((double) imageView.getWidth() / imageView.getHeight()) > aspectRatio) {
      width = (int) (height * aspectRatio);
    } else {
      height = (int) (width / aspectRatio);
    }
    return Bitmap.createScaledBitmap(originalBitmap, width, height, false);
  }

  private Bitmap rotateBitmap(Bitmap inputBitmap, InputStream imageData) throws IOException { //사진 회전 방지
    int orientation =
            new ExifInterface(imageData)
                    .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
    if (orientation == ExifInterface.ORIENTATION_NORMAL) {
      return inputBitmap;
    }
    Matrix matrix = new Matrix();
    switch (orientation) {
      case ExifInterface.ORIENTATION_ROTATE_90:
        matrix.postRotate(90);
        break;
      case ExifInterface.ORIENTATION_ROTATE_180:
        matrix.postRotate(180);
        break;
      case ExifInterface.ORIENTATION_ROTATE_270:
        matrix.postRotate(270);
        break;
      default:
        matrix.postRotate(0);
    }
    return Bitmap.createBitmap(
            inputBitmap, 0, 0, inputBitmap.getWidth(), inputBitmap.getHeight(), matrix, true);
  }

  /** Sets up the UI components for the static image demo. */
  private void setupStaticImageDemoUiComponents() {
    // The Intent to access gallery and read images as bitmap.
    imageGetter =
            registerForActivityResult(
                    new ActivityResultContracts.StartActivityForResult(),
                    result -> {
                      Intent resultIntent = result.getData();
                      if (resultIntent != null) {
                        if (result.getResultCode() == RESULT_OK) {
                          Bitmap bitmap = null;
                          try {
                            bitmap =
                                    downscaleBitmap(
                                            MediaStore.Images.Media.getBitmap(
                                                    this.getContentResolver(), resultIntent.getData()));
                          } catch (IOException e) {
                            Log.e(TAG, "Bitmap reading error:" + e);
                          }
                          try {
                            InputStream imageData =
                                    this.getContentResolver().openInputStream(resultIntent.getData());
                            bitmap = rotateBitmap(bitmap, imageData);
                          } catch (IOException e) {
                            Log.e(TAG, "Bitmap rotation error:" + e);
                          }
                          if (bitmap != null) {
                            hands.send(bitmap);
                          }
                        }
                      }
                    });

    try {
      imageView = new HandsResultImageView(this);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  /** Sets up the UI components for the live demo with camera input. */
  private void setupLiveDemoUiComponents() {
    Button startCameraButton = findViewById(R.id.button_start_camera);
    startCameraButton.setOnClickListener(
            v -> {
              if (inputSource == InputSource.CAMERA) {
                return;
              }
              stopCurrentPipeline();
              setupStreamingModePipeline(InputSource.CAMERA);
            });
  }

  /** Sets up core workflow for streaming mode. */
  private void setupStreamingModePipeline(InputSource inputSource) {
    this.inputSource = inputSource;
    // Initializes a new MediaPipe Hands solution instance in the streaming mode.
    hands =
            new Hands(
                    this,
                    HandsOptions.builder()
                            .setStaticImageMode(false)
                            .setMaxNumHands(2)
                            .setRunOnGpu(RUN_ON_GPU)
                            .build());
    hands.setErrorListener((message, e) -> Log.e(TAG, "MediaPipe Hands error:" + message));

    if (inputSource == InputSource.CAMERA) {
      cameraInput = new CameraInput(this);
      cameraInput.setNewFrameListener(textureFrame -> hands.send(textureFrame));
    } /*else if (inputSource == InputSource.VIDEO) {
      videoInput = new VideoInput(this);
      videoInput.setNewFrameListener(textureFrame -> hands.send(textureFrame));
    }*/

    // Initializes a new Gl surface view with a user-defined HandsResultGlRenderer.
    glSurfaceView =
            new SolutionGlSurfaceView<>(this, hands.getGlContext(), hands.getGlMajorVersion());
    glSurfaceView.setSolutionResultRenderer(new HandsResultGlRenderer());
    glSurfaceView.setRenderInputImage(true);

    hands.setResultListener(
            handsResult -> {
              //logWristLandmark(handsResult, /*showPixelValues=*/ false);
              glSurfaceView.setRenderData(handsResult);
              glSurfaceView.requestRender();
              makeAngle(handsResult);
            });

    // The runnable to start camera after the gl surface view is attached.
    // For video input source, videoInput.start() will be called when the video uri is available.
    if (inputSource == InputSource.CAMERA) {
      glSurfaceView.post(this::startCamera);
    }

    // Updates the preview layout.
    FrameLayout frameLayout = findViewById(R.id.preview_display_layout);
    imageView.setVisibility(View.GONE);
    frameLayout.removeAllViewsInLayout();
    frameLayout.addView(glSurfaceView);
    glSurfaceView.setVisibility(View.VISIBLE);
    frameLayout.requestLayout();
  }

  private void startCamera() {
    cameraInput.start(
            this,
            hands.getGlContext(),
            CameraInput.CameraFacing.FRONT,
            glSurfaceView.getWidth(),
            glSurfaceView.getHeight());
  }

  private void stopCurrentPipeline() {
    if (cameraInput != null) {
      cameraInput.setNewFrameListener(null);
      cameraInput.close();
    }
    if (glSurfaceView != null) {
      glSurfaceView.setVisibility(View.GONE);
    }
    if (hands != null) {
      hands.close();
    }
  }

  private void logWristLandmark(HandsResult result, boolean showPixelValues) {
    if (result.multiHandLandmarks().isEmpty()) {
      return;
    }
    NormalizedLandmark wristLandmark =
            result.multiHandLandmarks().get(0).getLandmarkList().get(HandLandmark.WRIST);
    // For Bitmaps, show the pixel values. For texture inputs, show the normalized coordinates.
    if (showPixelValues) {
      int width = result.inputBitmap().getWidth();
      int height = result.inputBitmap().getHeight();
      Log.i(
              TAG,
              String.format(
                      "MediaPipe Hand wrist coordinates (pixel values): x=%f, y=%f",
                      wristLandmark.getX() * width, wristLandmark.getY() * height));
    } else {
      Log.i(
              TAG,
              String.format(
                      "MediaPipe Hand wrist normalized coordinates (value range: [0, 1]): x=%f, y=%f",
                      wristLandmark.getX(), wristLandmark.getY()));
    }
    if (result.multiHandWorldLandmarks().isEmpty()) {
      return;
    }
    Landmark wristWorldLandmark =
            result.multiHandWorldLandmarks().get(0).getLandmarkList().get(HandLandmark.WRIST);
    Log.i(
            TAG,
            String.format(
                    "MediaPipe Hand wrist world coordinates (in meters with the origin at the hand's"
                            + " approximate geometric center): x=%f m, y=%f m, z=%f m",
                    wristWorldLandmark.getX(), wristWorldLandmark.getY(), wristWorldLandmark.getZ()));
  }

  private void makeAngle(HandsResult handsresult) { //각도구하는 함수

    if (handsresult == null) {
      return;
    }
    float[][] vertices = new float[21][3];

    int k = 0;
    int numHands = handsresult.multiHandLandmarks().size();
    for (int i = 0; i < numHands; ++i) {
      for (LandmarkProto.NormalizedLandmark pointLandmark : handsresult.multiHandLandmarks().get(i).getLandmarkList()) {
        vertices[k][0] = pointLandmark.getX();
        vertices[k][1] = pointLandmark.getY();
        vertices[k][2] = pointLandmark.getZ();
        k += 1;
      }
    }

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
      for (int l = 0; l < v[i].length; l++) {
        v[i][l] = powSum[i]/powCal[i]*v[i][l];
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

    float max = output[0][0]; //확률
    int maxIndex = 0;

    for (int i=0 ; i < output[0].length; i++) {
      if (output[0][i] > max) {
        max = output[0][i];
        maxIndex = i;
      }
      data = gesture[maxIndex];
    }
    TextView tv = findViewById(R.id.text_view);
    tv.setText(data);
    Toast.makeText(this,"정확도 : %f"+max,Toast.LENGTH_SHORT).show(); //정확도값 토스트로 띄움


    myRef = database.getReference(String.valueOf(data));
    myRef.addValueEventListener(new ValueEventListener() {
      String[] word_list = new String[6];

      public void onDataChange(DataSnapshot dataSnapshot) {
        for (DataSnapshot snapshot : dataSnapshot.getChildren()) {
          for (DataSnapshot snapshot2 : snapshot.getChildren()) {
            Log.i("osslog", snapshot2.getValue().toString());
            for (int k = 0 ; k<6 ;k++){
              word_list[k] = snapshot2.getValue().toString();
            }
          }
        }

        TextView tv2 = findViewById(R.id.text_view2);
        Button bt1 = findViewById(R.id.button_voca1);
        Button bt2 = findViewById(R.id.button_voca2);
        Button bt3 = findViewById(R.id.button_voca3);
        Button bt4 = findViewById(R.id.button_voca4);
        Button bt5 = findViewById(R.id.button_voca5);
        Button bt6 = findViewById(R.id.button_voca6);

        bt1.setText(word_list[0]); //btn text 바꿔줌
        bt2.setText(word_list[1]);
        bt3.setText(word_list[2]);
        bt4.setText(word_list[3]);
        bt5.setText(word_list[4]);
        bt6.setText(word_list[5]);

        bt1.setOnClickListener(new Button.OnClickListener(){ //자동완성 bt1 눌렀을 때 text 변경
          public void onClick(View v){
            tv2.append(word_list[0]);
          }
        });
        bt2.setOnClickListener(new Button.OnClickListener(){ //자동완성 bt2 눌렀을 때 text 변경
          public void onClick(View v){
            tv2.setText(word_list[1]);
          }
        });
        bt3.setOnClickListener(new Button.OnClickListener(){ //자동완성 bt3 눌렀을 때 text 변경
          public void onClick(View v){
            tv2.setText(word_list[2]);
          }
        });
        bt4.setOnClickListener(new Button.OnClickListener(){ //자동완성 bt4 눌렀을 때 text 변경
          public void onClick(View v){
            tv2.setText(word_list[3]);
          }
        });
        bt5.setOnClickListener(new Button.OnClickListener(){ //자동완성 bt5 눌렀을 때 text 변경
          public void onClick(View v){
            tv2.setText(word_list[4]);
          }
        });
        bt6.setOnClickListener(new Button.OnClickListener(){ //자동완성 bt6 눌렀을 때 text 변경
          public void onClick(View v){
            tv2.setText(word_list[5]);
          }
        });
      } //onDataChange
      @Override
      public void onCancelled(DatabaseError databaseError) {
        //Toast.makeText(HandsResultGlRenderer.this,"error: " + databaseError.getMessage(), Toast.LENGTH_SHORT).show();
      }

    }); //myRef.addValueEventListener

    // 완료 버튼 클릭시 액티비티 전환
    Button bt_com = findViewById(R.id.button_complete);
    bt_com.setOnClickListener(new View.OnClickListener(){

      @Override
      public void onClick(View view){
        Intent intent = new Intent(getApplicationContext(), resultActivity.class);
        startActivity(intent);
      }
    });
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