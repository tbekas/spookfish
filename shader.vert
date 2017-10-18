#version 130
in vec4 position;
in vec2 tex_coord;
out vec2 colorCoord;
void main()
{
    gl_Position = position;
    colorCoord = tex_coord;
}
