package com.example.vibro2;

import androidx.appcompat.app.AppCompatActivity;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.os.Vibrator;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Timer;
import java.util.TimerTask;

public class Application extends AppCompatActivity implements SensorEventListener {

    public boolean isVibrate;
    public boolean needCalculate;
    public Button start;
    public PyObject pyobj;
    public Vibrator vibe;
    //Store data
    File storageData;
    FileWriter fw;
    public String path;
    public String fileName;
    //Variable for data and status display
    public float data1 = 0;
    public float data2 = 0;
    public float data3 = 0;
    public long time_ms;
    //Accelerometer Sensor
    public Sensor accelerometer;
    public SensorManager accelerometerManager;    @Override
    public void onSensorChanged(SensorEvent event) {
        time_ms = System.currentTimeMillis() + (event.timestamp - SystemClock.elapsedRealtimeNanos()) / 1000000L;
        data1 = event.values[0];
        data2 = event.values[1];
        data3 = event.values[2];
//        Log.e(time_ms+"", isVibrate+"");
        if(isVibrate){
//                final_string=final_string.concat(time_ms+","+data1+","+data2+","+data3+"\n");
            try{
                fw = new FileWriter(storageData,true);
                fw.append(time_ms+","+data1+","+data2+","+data3+"\n");
                fw.close();
            }
            catch(IOException e){
                Log.e("Failure","Fail to create file writer");
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    @Override
    public void onPointerCaptureChanged(boolean hasCapture) {

    }

    @Override
    protected void onResume() {
        super.onResume();
        accelerometerManager.registerListener(this, accelerometer, 2520);
    }

    @Override
    protected void onPause() {
        super.onPause();
        accelerometerManager.unregisterListener(this);
    }

    //testTimer run data collection in 20Hz
    public class testTimer {
        Timer timer;
        int delay;
        public testTimer(int d) {
            delay = d;
            timer = new Timer();
            Log.e("delay",delay+"");
            timer.schedule(new Application.testTimer.RemindTask(), delay);
        }

        class RemindTask extends TimerTask {
            public void run() {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        isVibrate = false;
                        TextView show = findViewById(R.id.info);
                        show.setText("Wait");
//                        try{
//                            PrintWriter pw = new PrintWriter(storageData);
//                            pw.print(final_string);
//                            pw.close();
//                        }
//                        catch(IOException e){
//                            Log.e("Failure","File printer fail");
//                        }


                    }
                });
            }
        }    }
    public class delayTimer {
        Timer timer;
        int delay;
        public delayTimer(int d) {
            delay = d;
            timer = new Timer();
            Log.e("delay",delay+"");
            timer.schedule(new Application.delayTimer.RemindTask(), delay);
        }

        class RemindTask extends TimerTask {
            public void run() {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        TextView show = findViewById(R.id.info);
                        show.setText("Put it now");
//                        recorder.stop();
//                        recorder.release();
//                        recorder = null;
//                        try{
//                            PrintWriter pw = new PrintWriter(storageData);
//                            pw.print(final_string);
//                            pw.close();
//                        }
//                        catch(IOException e){
//                            Log.e("Failure","File printer fail");
//                        }
                    }
                });
            }
        }    }
    //Caculate the weight with python code
    public class showWeightTimer {
        Timer timer;
        int delay;
        public showWeightTimer(int d) {
            delay = d;
            timer = new Timer();
            Log.e("delay",delay+"");
            timer.schedule(new RemindTask(), delay);

        }

        class RemindTask extends TimerTask {
            public void run() {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Log.i("path: ",fileName);
                        PyObject obj = pyobj.callAttr("main",fileName);
                        TextView show = findViewById(R.id.info);
                        String result = obj.toString();
                        show.setText("Result: " + result);

                    }
                });
            }
        }    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //Setting accelerometer
        accelerometerManager = (SensorManager)getSystemService(this.SENSOR_SERVICE);
        accelerometer = accelerometerManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        vibe = (Vibrator) getSystemService(this.VIBRATOR_SERVICE);
        path = this.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath();
        setContentView(R.layout.activity_application);
        if(!Python.isStarted()){
            Python.start(new AndroidPlatform(this));
        }
        Python py = Python.getInstance();
        pyobj = py.getModule("application");
        isVibrate = false;
        needCalculate = false;
        start = this.findViewById(R.id.vibrate);
        start.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                //Set vibration parameter
                TextView show = findViewById(R.id.info);
                show.setText("Wait");
                isVibrate = true;
                //Create a new csv to store the data
                fileName = path + "/" + time_ms + "_test.csv";
                storageData = new File(fileName);
                try{
                    storageData.createNewFile();
                }
                catch(IOException e){
                    System.out.print("no exist");
                }
                try{
                    fw = new FileWriter(storageData,true);
                    fw.append("time_tick,acc_X_value,acc_Y_value,acc_Z_value\n");
                    fw.close();
                }
                catch(IOException e){
                    Log.e("Failure","Fail to create file writer");
                }
                //Start vibration
                long[] pattern = {0,10000};

                vibe.vibrate(pattern,-1);
                Log.e("pattern", pattern.toString());
                new Application.testTimer(10010);
                new Application.delayTimer(5000);
                //Start vibration
                new showWeightTimer(12010);
                needCalculate = true;

            }
        });
    }

}