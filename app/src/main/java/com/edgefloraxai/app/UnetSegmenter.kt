package com.edgefloraxai.app

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp

data class SegmentationResult(
    val diseaseAreaPercent: Float,
    val severityPercent: Float,
    val severityBand: String,
    val mask: Array<FloatArray>
)

class UnetSegmenter(
    private val context: Context
) {

    private val interpreter =
        Interpreter(
            loadModelFile("unet_float32.tflite")
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

    fun segment(
        bitmap: Bitmap,
        diseaseName: String
    ): SegmentationResult {

        val resized =
            Bitmap.createScaledBitmap(
                bitmap,
                256,
                256,
                true
            )

        val input =
            ByteBuffer.allocateDirect(
                256 * 256 * 3 * 4
            ).order(
                ByteOrder.nativeOrder()
            )

        for (y in 0 until 256) {

            for (x in 0 until 256) {

                val pixel =
                    resized.getPixel(x, y)

                val r =
                    ((pixel shr 16) and 0xFF) / 255.0f

                val g =
                    ((pixel shr 8) and 0xFF) / 255.0f

                val b =
                    (pixel and 0xFF) / 255.0f

                input.putFloat(r)
                input.putFloat(g)
                input.putFloat(b)
            }
        }

        input.rewind()

        // U-Net output shape is:
        // [1, 256, 256, 1]

        val output =
            Array(1) {
                Array(256) {
                    Array(256) {
                        FloatArray(1)
                    }
                }
            }

        interpreter.run(
            input,
            output
        )

        var diseasePixels = 0

        val mask =
            Array(256) {
                FloatArray(256)
            }

        for (y in 0 until 256) {

            for (x in 0 until 256) {

                val logit =
                    output[0][y][x][0]

                val probability =
                    sigmoid(logit)

                mask[y][x] =
                    probability

                if (probability >= 0.5f) {
                    diseasePixels++
                }
            }
        }

        val diseaseAreaPercent =
            diseasePixels.toFloat() /
                    (256f * 256f) *
                    100f

        val severityPercent =
            if (diseaseName == "Healthy") {

                0f

            } else {

                minOf(
                    diseaseAreaPercent /
                            50f *
                            100f,
                    100f
                )
            }

        val severityBand =
            when {

                diseaseName == "Healthy" ->
                    "Healthy"

                severityPercent == 0f ->
                    "Very Low / Uncertain"

                severityPercent < 10f ->
                    "Low"

                severityPercent < 30f ->
                    "Moderate"

                severityPercent < 60f ->
                    "High"

                else ->
                    "Severe"
            }

        if (resized !== bitmap) {
            resized.recycle()
        }

        return SegmentationResult(
            diseaseAreaPercent =
                diseaseAreaPercent,

            severityPercent =
                severityPercent,

            severityBand =
                severityBand,

            mask =
                mask
        )
    }

    private fun sigmoid(
        value: Float
    ): Float {

        return if (value >= 0f) {

            val z =
                exp(-value.toDouble())

            (1.0 / (1.0 + z)).toFloat()

        } else {

            val z =
                exp(value.toDouble())

            (z / (1.0 + z)).toFloat()
        }
    }

    fun close() {
        interpreter.close()
    }
}