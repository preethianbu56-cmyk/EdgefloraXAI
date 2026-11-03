package com.edgefloraxai.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.min

data class YoloResult(
    val classId: Int,
    val className: String,
    val confidence: Float
)

class YoloDetector(private val context: Context) {

    private val classNames = listOf(
        "Chrysanthemum",
        "Hibiscus",
        "Jasmine",
        "Orchid",
        "Rose",
        "Sunflower"
    )

    private val interpreter =
        Interpreter(loadModelFile("yolo_best.tflite"))

    private fun loadModelFile(fileName: String): ByteBuffer {

        val fd =
            context.assets.openFd(fileName)

        val inputStream =
            FileInputStream(fd.fileDescriptor)

        return inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY,
            fd.startOffset,
            fd.declaredLength
        ).order(ByteOrder.nativeOrder())
    }

    private fun letterbox(bitmap: Bitmap): Bitmap {

        val targetSize = 640

        val scale = min(
            targetSize.toFloat() / bitmap.width.toFloat(),
            targetSize.toFloat() / bitmap.height.toFloat()
        )

        val newWidth =
            (bitmap.width * scale).toInt()

        val newHeight =
            (bitmap.height * scale).toInt()

        val resized =
            Bitmap.createScaledBitmap(
                bitmap,
                newWidth,
                newHeight,
                true
            )

        val output =
            Bitmap.createBitmap(
                targetSize,
                targetSize,
                Bitmap.Config.ARGB_8888
            )

        val canvas =
            Canvas(output)

        canvas.drawColor(
            Color.rgb(114, 114, 114)
        )

        val left =
            (targetSize - newWidth) / 2f

        val top =
            (targetSize - newHeight) / 2f

        canvas.drawBitmap(
            resized,
            left,
            top,
            null
        )

        if (resized !== bitmap) {
            resized.recycle()
        }

        return output
    }

    fun detect(bitmap: Bitmap): List<YoloResult> {

        val image =
            letterbox(bitmap)

        /*
         * Model input is:
         *
         * [1, 3, 640, 640]
         *
         * Therefore the data must be stored as:
         *
         * all RED pixels
         * then all GREEN pixels
         * then all BLUE pixels
         */

        val input =
            ByteBuffer.allocateDirect(
                1 * 3 * 640 * 640 * 4
            ).order(ByteOrder.nativeOrder())

        // RED channel
        for (y in 0 until 640) {
            for (x in 0 until 640) {

                val pixel =
                    image.getPixel(x, y)

                input.putFloat(
                    Color.red(pixel) / 255.0f
                )
            }
        }

        // GREEN channel
        for (y in 0 until 640) {
            for (x in 0 until 640) {

                val pixel =
                    image.getPixel(x, y)

                input.putFloat(
                    Color.green(pixel) / 255.0f
                )
            }
        }

        // BLUE channel
        for (y in 0 until 640) {
            for (x in 0 until 640) {

                val pixel =
                    image.getPixel(x, y)

                input.putFloat(
                    Color.blue(pixel) / 255.0f
                )
            }
        }

        input.rewind()

        /*
         * YOLO output:
         *
         * [1, 10, 8400]
         *
         * 0-3  = box values
         * 4-9  = six class scores
         */

        val output =
            Array(1) {
                Array(10) {
                    FloatArray(8400)
                }
            }

        interpreter.run(
            input,
            output
        )

        var bestClassId = -1
        var bestConfidence = 0.0f

        for (i in 0 until 8400) {

            for (classId in 0 until 6) {

                val confidence =
                    output[0][4 + classId][i]

                if (confidence > bestConfidence) {

                    bestConfidence =
                        confidence

                    bestClassId =
                        classId
                }
            }
        }

        image.recycle()

        if (bestClassId == -1) {
            return emptyList()
        }

        return listOf(
            YoloResult(
                classId = bestClassId,
                className = classNames[bestClassId],
                confidence = bestConfidence
            )
        )
    }

    fun getModelInfo(): String {

        val inputTensor =
            interpreter.getInputTensor(0)

        val outputTensor =
            interpreter.getOutputTensor(0)

        return """
INPUT SHAPE:
${inputTensor.shape().contentToString()}

INPUT TYPE:
${inputTensor.dataType()}

OUTPUT SHAPE:
${outputTensor.shape().contentToString()}

OUTPUT TYPE:
${outputTensor.dataType()}

INPUT COUNT:
${interpreter.inputTensorCount}

OUTPUT COUNT:
${interpreter.outputTensorCount}
        """.trimIndent()
    }

    fun close() {
        interpreter.close()
    }
}