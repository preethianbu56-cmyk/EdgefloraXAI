package com.edgefloraxai.app

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp

data class DiseaseResult(
    val classId: Int,
    val className: String,
    val confidence: Float
)

class MobileVitClassifier(
    private val context: Context
) {

    private val classNames = listOf(
        "Anthracnose",
        "Bacterial_leaf",
        "Blackspot",
        "Chlorosis",
        "Gray_mold",
        "Healthy",
        "PowderyMildew",
        "Septoria_leaf",
        "Soft",
        "Wilted",
        "blight",
        "scorch"
    )

    private val interpreter =
        Interpreter(
            loadModelFile("mobilevit_s_float32.tflite")
        )

    private fun loadModelFile(
        fileName: String
    ): ByteBuffer {

        val fd =
            context.assets.openFd(fileName)

        val inputStream =
            FileInputStream(
                fd.fileDescriptor
            )

        return inputStream.channel
            .map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength
            )
            .order(ByteOrder.nativeOrder())
    }

    fun classify(
        bitmap: Bitmap
    ): DiseaseResult {

        val resized =
            Bitmap.createScaledBitmap(
                bitmap,
                224,
                224,
                true
            )

        val input =
            ByteBuffer.allocateDirect(
                224 * 224 * 3 * 4
            ).order(
                ByteOrder.nativeOrder()
            )

        for (y in 0 until 224) {

            for (x in 0 until 224) {

                val pixel =
                    resized.getPixel(
                        x,
                        y
                    )

                val r =
                    (pixel shr 16) and 0xFF

                val g =
                    (pixel shr 8) and 0xFF

                val b =
                    pixel and 0xFF

                input.putFloat(
                    (r / 255.0f - 0.485f) / 0.229f
                )

                input.putFloat(
                    (g / 255.0f - 0.456f) / 0.224f
                )

                input.putFloat(
                    (b / 255.0f - 0.406f) / 0.225f
                )
            }
        }

        input.rewind()

        val output =
            Array(1) {
                FloatArray(12)
            }

        interpreter.run(
            input,
            output
        )

        val probabilities =
            softmax(output[0])

        var bestClassId = 0

        for (i in 1 until 12) {

            if (
                probabilities[i] >
                probabilities[bestClassId]
            ) {
                bestClassId = i
            }
        }

        val result =
            DiseaseResult(
                classId = bestClassId,
                className =
                    classNames[bestClassId],
                confidence =
                    probabilities[bestClassId]
            )

        if (resized !== bitmap) {
            resized.recycle()
        }

        return result
    }

    private fun softmax(
        values: FloatArray
    ): FloatArray {

        var maxValue =
            values[0]

        for (value in values) {

            if (value > maxValue) {
                maxValue = value
            }
        }

        val expValues =
            FloatArray(values.size)

        var sum = 0.0

        for (i in values.indices) {

            expValues[i] =
                exp(
                    (
                            values[i] -
                                    maxValue
                            ).toDouble()
                ).toFloat()

            sum +=
                expValues[i].toDouble()
        }

        for (i in expValues.indices) {

            expValues[i] =
                (
                        expValues[i] /
                                sum
                        ).toFloat()
        }

        return expValues
    }

    fun close() {
        interpreter.close()
    }
}