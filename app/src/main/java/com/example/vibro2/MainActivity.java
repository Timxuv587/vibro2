package com.example.vibro2;

import android.os.Bundle;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.Manifest;
import android.content.pm.PackageManager;
import java.io.IOException;
import java.io.File;
import java.io.FileWriter;
import android.net.Uri;

import java.io.PrintWriter;
import java.util.Timer;
import java.util.TimerTask;

import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.content.Intent;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Button;
import android.os.Vibrator;
import android.os.VibrationEffect;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.media.MediaRecorder;
import android.os.Environment;



public class MainActivity extends AppCompatActivity implements SensorEventListener{

    //Variable for control vibration
    public boolean isVibrate;
    public long time_ms;
    public Vibrator vibe;
    public Button start20;
    public Button stop;
    public Button frequencyTest;
    public EditText cycle;
    public EditText r;
    public EditText repeat;
    public EditText weight;
    public EditText ID;
    public EditText fruit;
    public EditText delay;
    //Variable for data and status display
    public float data1 = 0;
    public float data2 = 0;
    public float data3 = 0;
    public TextView dataDisplay1;
    public TextView dataDisplay2;
    public TextView dataDisplay3;


    //Accelerometer Sensor
    public Sensor accelerometer;
    public SensorManager accelerometerManager;
    //gyroscope Sensor
    public Sensor gyroscope;
    public SensorManager gyroscopeManager;

    //Microphone function
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private boolean permissionToRecordAccepted = false;
    private String [] permissions = {Manifest.permission.RECORD_AUDIO};
    private MediaRecorder recorder = null;

    //Store data
    File storageData;
    FileWriter fw;
    String path;


//    @Override
//    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
//        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
//        switch (requestCode){
//            case REQUEST_RECORD_AUDIO_PERMISSION:
//                permissionToRecordAccepted  = grantResults[0] == PackageManager.PERMISSION_GRANTED;
//                break;
//        }
//        if (!permissionToRecordAccepted ) finish();
//
//    }

    //testTimer run data collection in 20Hz
    public class testTimer {
        Timer timer;
        int delay;
        public testTimer(int d) {
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
                        isVibrate = false;
                        TextView show = findViewById(R.id.textView4);
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
            timer.schedule(new RemindTask(), delay);

        }

        class RemindTask extends TimerTask {
            public void run() {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        TextView show = findViewById(R.id.textView4);
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
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        path = this.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath();
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        //Setting input
        cycle = findViewById(R.id.inputCycle);
        cycle.setText("100");
        r = findViewById(R.id.inputRatio);
        r.setText("0.5");
        repeat = findViewById(R.id.inputRepeat);
        repeat.setText("1");
        weight = findViewById(R.id.inputWeight);
        ID = findViewById(R.id.inputID);
        fruit = findViewById(R.id.inputFruit);
        delay = findViewById(R.id.inputDelay);
        //Setting File writing function
        Log.e("Target",path);
        //Setting Button function
        FloatingActionButton fab = findViewById(R.id.fab);
        vibe = (Vibrator) getSystemService(this.VIBRATOR_SERVICE);
        start20 = findViewById(R.id.button1);
        stop = findViewById(R.id.button3);
        frequencyTest = findViewById(R.id.button4);
        start20.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                //Set vibration parameter
                isVibrate = true;
                int c = Integer.valueOf(cycle.getText().toString());
                Float ratio = Float.parseFloat(r.getText().toString());
                int rep = Integer.valueOf(repeat.getText().toString())*2;
                long[] pattern = new long[rep+1];
                pattern[0] = Integer.valueOf(delay.getText().toString());
                int vib = (int)(c*ratio);
                int space = c - vib;
                for(int i=1; i < rep; i++)
                    if(i % 2 == 0)
                        pattern[i] = space;
                    else
                        pattern[i] = vib;

                //Create a new csv to store the data
                storageData = new File(path + "/" + fruit.getText().toString() + "_"+ID.getText().toString() + "_"+weight.getText().toString() + "_"+time_ms+"_"+delay.getText().toString()+"_"+c+"_"+ratio+"_"+rep+".csv");
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
                vibe.vibrate(pattern,-1);
                Log.e("pattern", pattern.toString());
                new testTimer(c*rep/2+10);
                new delayTimer(2000);
//                new delayTimer((c*rep/2)/2);
//                //Start recording
//                recorder = new MediaRecorder();
//                recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
//                recorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
//                recorder.setOutputFile(path + "/" + fruit.getText().toString() + "_"+ID.getText().toString() + "_"+weight.getText().toString() + "_"+time_ms+"_"+delay.getText().toString()+"_"+c+"_"+ratio+"_"+rep+".csv");
//                recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
//
//                try {
//                    recorder.prepare();
//                } catch (IOException e) {
//                    Log.e("Recording", "prepare() failed");
//                }
//
//                recorder.start();
            }
        });
        stop.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Vibrator tempt = vibe;
                vibe.cancel();
                vibe = tempt;
            }
        });
        frequencyTest.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                //Set vibration parameter
                isVibrate = true;
                int c = Integer.valueOf(cycle.getText().toString());
                Float ratio = Float.parseFloat("0");
                int rep = Integer.valueOf(repeat.getText().toString())*2;
                long[] pattern = new long[rep+1];
                pattern[0] = Integer.valueOf(delay.getText().toString());
                for(int i=1; i < rep; i++){
                    if(i % 2 == 0)
                    {
                        double tempt = (i/2)*((double)c/((double)rep/2));
                        int vib = (int)tempt;
                        pattern[i] = vib;
                    }
                    else
                    {
                        double tempt =  (i-1)/2*((double)c/((double)rep/2));
                        int vib = (int)tempt;
                        int space = c - vib;
                        pattern[i] = space;
                    }

                }


                //Create a new csv to store the data
                storageData = new File(path + "/" + fruit.getText().toString() + "_"+ID.getText().toString() + "_"+weight.getText().toString() + "_"+time_ms+"_"+delay.getText().toString()+"_"+c+"_"+ratio+"_"+rep+".csv");
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
                vibe.vibrate(pattern,-1);
                Log.e("pattern", pattern.toString());
                new testTimer(c*rep/2+10);
            }
        });
        fab.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                sendEmail("1430415000@qq.com",storageData);
            }
        });

        //Setting accelerometer
        accelerometerManager = (SensorManager)getSystemService(this.SENSOR_SERVICE);
        accelerometer = accelerometerManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        dataDisplay1 = findViewById(R.id.visualize1);
        dataDisplay2 = findViewById(R.id.visualize2);
        dataDisplay3 = findViewById(R.id.visualize3);

    }

    public final void onSensorChanged(SensorEvent event) {
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
        dataDisplay1.setText(data1+"");
        dataDisplay2.setText(data2+"");
        dataDisplay3.setText(data3+"");
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }
        return super.onOptionsItemSelected(item);
    }
    @Override
    public final void onAccuracyChanged(Sensor sensor, int accuracy) {
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

    protected void sendEmail(String address, File storageData) {
        String TO[] = {address};
        Uri path = Uri.fromFile(storageData);
        Intent emailIntent = new Intent(Intent.ACTION_SEND);
        emailIntent.setData(Uri.parse("mailto:"));
        emailIntent.setType("vnd.android.cursor.dir/email");
        emailIntent.putExtra(Intent.EXTRA_STREAM, path);
        startActivity(Intent.createChooser(emailIntent, "Send mail..."));
    }
}
