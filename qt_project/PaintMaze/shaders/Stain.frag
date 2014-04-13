#version 120
#extension GL_EXT_gpu_shader4 : enable
uniform sampler2D qt_Texture0;
varying vec4 qt_TexCoord0;
uniform vec4 color;

/**
  * Dessine une tache sur un quad.
  */
void main(void)
{
    gl_FragColor = color;
}
