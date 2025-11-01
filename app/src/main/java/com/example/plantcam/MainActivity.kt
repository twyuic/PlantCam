package com.example.plantcam

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var tflite: Interpreter
    private lateinit var classNames: List<String>


    private val knownLabels = setOf(
        "Corn___healthy",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry___healthy",
        "Grape___healthy",
        "Peach___healthy",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Strawberry___healthy",
        "Tomato___healthy"
    )


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)

        // 載入模型和 class_names
        tflite = Interpreter(loadModelFile("plant_model.tflite"))
        classNames = assets.open("class_names.txt").bufferedReader().readLines()

        // 請求權限
        requestCameraPermission()
    }

    private fun requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.CAMERA), 0)
        } else {
            startCamera()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 0 && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            // ImageAnalysis 用於即時辨識
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalyzer.setAnalyzer(ContextCompat.getMainExecutor(this)) { imageProxy ->
                val bitmap = previewView.bitmap
                bitmap?.let {
                    val (label, confidence) = predict(it)

                    // 顯示文字：包含信心值百分比
                    val displayText = if (label == "無法辨識") {
                        "無法辨識"
                    } else {
                        "$label"
                    }

                    // 無法辨識 → 紅色，有辨識 → 綠色
                    val color = if (label == "無法辨識") {Color.RED}
                    else if(label in knownLabels) {
                        Color.GREEN}
                    else {
                        Color.RED
                    }

                    overlayView.setLabel(displayText, color)
                }
                imageProxy.close()
            }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e("PlantCam", "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // TFLite 預測，回傳 label + 信心值
    private fun predict(bitmap: Bitmap): Pair<String, Float> {
        // 將 bitmap resize 成模型輸入大小
        val input = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val byteBuffer = convertBitmapToByteBuffer(input)

        val output = Array(1) { FloatArray(classNames.size) }
        tflite.run(byteBuffer, output)

        // 找最大機率與索引
        val maxIndex = output[0].indices.maxByOrNull { output[0][it] } ?: 0
        val confidence = output[0][maxIndex]

        // 設定一個閾值 (例如 0.6)
        val threshold = 0.6f
        val label = if (confidence >= threshold) {
            classNames[maxIndex]
        } else {
            "無法辨識"
        }

        return Pair(label, confidence)
    }

    private fun loadModelFile(filename: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val channel = inputStream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(1 * 224 * 224 * 3 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(224 * 224)
        bitmap.getPixels(intValues, 0, 224, 0, 0, 224, 224)
        for (pixelValue in intValues) {
            val r = (pixelValue shr 16 and 0xFF) / 255f
            val g = (pixelValue shr 8 and 0xFF) / 255f
            val b = (pixelValue and 0xFF) / 255f
            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
        }
        return byteBuffer
    }
}
