#version 130
in vec2 colorCoord;

uniform sampler2D colorTexture;
uniform float brightness;

out vec4 outputColor;

void main()
{
    vec4 color = texture(colorTexture, colorCoord);
    outputColor = color * brightness;
}
