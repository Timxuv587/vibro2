package com.example.vibro2;

import android.os.Bundle;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

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
import android.os.Environment;


public class MainActivity extends AppCompatActivity implements SensorEventListener{

    //Variable for control vibration
    public Vibrator vibe;
    public Button start20;
    public Button delete;
    public Button stop;
    public EditText cycle;
    public EditText r;
    public EditText repeat;
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

    //Store data
    File storageData;
    FileWriter fw;
    String path;


    //testTimer run data collection in 20Hz
    public class testTimer {
        Timer timer;
        int time = 0;
        public testTimer() {
            timer = new Timer();
            timer.schedule(new RemindTask(), 20, 100);
        }

        class RemindTask extends TimerTask {
            public void run() {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        time ++;
                        if(fw != null)
                            try{
                                fw.append(data1 + "," + data2 + "," + data3 + '\n');
                            }
                            catch(IOException e){
                                Log.e("Failuer","no exist");
                            }
                        dataDisplay1.setText(data1+"");
                        dataDisplay2.setText(data2+"");
                        dataDisplay3.setText(data3+"");
                    }
                });
            }
        }
    }
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

        //Setting File writing function
        Log.e("Target",path);
        storageData = new File(path + "/data.csv");
        //new testTimer();
        try{
            storageData.createNewFile();
        }
        catch(IOException e){
            System.out.print("no exist");
        }
        try{
            fw = new FileWriter(storageData, true) ;
        }
        catch(IOException e){
            Log.e("Failure","fail to create writing fail");
        }

        //Setting Button function
        FloatingActionButton fab = findViewById(R.id.fab);
        vibe = (Vibrator) getSystemService(this.VIBRATOR_SERVICE);
        start20 = findViewById(R.id.button1);
        delete = findViewById(R.id.button2);
        stop = findViewById(R.id.button3);
        start20.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                int c = Integer.valueOf(cycle.getText().toString());
                Float ratio = Float.parseFloat(r.getText().toString());
                int rep = Integer.valueOf(repeat.getText().toString())*2;
                long[] pattern = new long[rep];
                int vib = (int)(c*ratio);
                int space = c - vib;
                for(int i=0; i < rep; i++)
                    if(i % 2 == 0)
                        pattern[i] = space;
                    else
                        pattern[i] = vib;
                vibe.vibrate(pattern,-1);

            }
        });
        stop.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Vibrator tempt = vibe;
                vibe.cancel();
                vibe = tempt;
            }
        });
        fab.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                sendEmail("1430415000@qq.com",storageData);
            }
        });
        delete.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                try{
                    PrintWriter writer = new PrintWriter(storageData);
                    writer.print("");
                    writer.close();
                }
                catch(java.io.FileNotFoundException e){
                    Log.e("Failure","File not found");
                }
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
        long time_ms = System.currentTimeMillis() + (event.timestamp - SystemClock.elapsedRealtimeNanos()) / 1000000L;
        data1 = event.values[0];
        data2 = event.values[1];
        data3 = event.values[2];
        if(fw != null)
            try{
                fw.append(time_ms+","+data1 + "," + data2 + "," + data3 + '\n');
            }
            catch(IOException e){
                Log.e("Failure","no exist");
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
        accelerometerManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);
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
