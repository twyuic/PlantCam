package com.example.plantcam

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context, attrs: AttributeSet): View(context, attrs) {

    private var label: String? = null
    private val paint = Paint().apply {
        color = Color.RED
        textSize = 80f
        typeface = Typeface.DEFAULT_BOLD
    }

    private var textColor: Int = Color.GREEN

    fun setLabel(label: String, color: Int = Color.GREEN) {
        this.label = label
        this.textColor = color
        paint.color = color
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        label?.let {
            canvas.drawText(it, 50f, 100f, paint)
        }
    }
}
