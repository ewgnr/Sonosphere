#version 330 core

const int MAX_MARCHING_STEPS = 255;
const float MIN_DIST = 0.0;
const float MAX_DIST = 100.0;
const float EPSILON = 0.0001;
const float PI = 3.1415926535897932384626433832795; // there is no PI defined in GLSL?

const int jointCount = 24; // 4 arms
const int edgeCount = 23; // 4 arms 
const int objectCount = 27; // 27

uniform float iGlobalTime;
vec4 vectorTime = vec4(iGlobalTime / 20.0, iGlobalTime, iGlobalTime * iGlobalTime, iGlobalTime * iGlobalTime * iGlobalTime); // this is for some of the fractals

in vec2 fragCoord;
out vec4 fragColor;

// camera settings
uniform vec3 camPosition;
uniform float camAngle;

// light settings
uniform vec3 lightPosition;
uniform float shadowSmooth;
uniform float shadowStrength;

// background color
uniform vec3 bgColor;
uniform vec3 bgOcclusionColor;

// fog limits
uniform float fog_min_dist;
uniform float fog_max_dist;

//ray settings
uniform float rayRotation;
uniform float rayWiggle;

// surface struct

struct Surface 
{
    vec3 color;
    float ambientScale;
    float diffuseScale;
    float specularScale;
    float specularPow;
    float occlusionScale;
    float occlusionRange;
    float occlusionResolution;
    vec3 occlusionColor; 
    float signedDistance;
};

// skeleton joint settings
uniform vec3 jointColor;
uniform float jointAmbientScale;
uniform float jointDiffuseScale;
uniform float jointSpecularScale;
uniform float jointSpecularPow;
uniform float jointOcclusionScale;
uniform float jointOcclusionRange;
uniform float jointOcclusionResolution;
uniform vec3 jointOcclusionColor;

uniform float jointPrimitives[jointCount];
uniform mat4 jointTransforms[jointCount];
uniform vec3 jointSizes[jointCount];
uniform float jointRoundings[jointCount];
uniform float jointSmoothings[jointCount];

// skeleton edge settings
uniform vec3 edgeColor;
uniform float edgeAmbientScale;
uniform float edgeDiffuseScale;
uniform float edgeSpecularScale;
uniform float edgeSpecularPow;
uniform float edgeOcclusionScale;
uniform float edgeOcclusionRange;
uniform float edgeOcclusionResolution;
uniform vec3 edgeOcclusionColor;

uniform float edgePrimitives[edgeCount];
uniform mat4 edgeTransforms[edgeCount];
uniform float edgeLengths[edgeCount];
uniform vec3 edgeSizes[edgeCount];
uniform float edgeRoundings[edgeCount];
uniform float edgeSmoothings[edgeCount];

// object settings
uniform vec3 objectColors[objectCount];
uniform float objectAmbientScales[objectCount];
uniform float objectDiffuseScales[objectCount];
uniform float objectSpecularScales[objectCount];
uniform float objectSpecularPows[objectCount];
uniform float objectOcclusionScales[objectCount];
uniform float objectOcclusionRanges[objectCount];
uniform float objectOcclusionResolutions[objectCount];
uniform vec3 objectOcclusionColors[objectCount];

uniform vec3 objectFrequencies[objectCount];
uniform vec3 objectAmplitudes[objectCount];
uniform vec3 objectPhases[objectCount];

uniform float objectPrimitives[objectCount];
uniform mat4 objectTransforms[objectCount];
uniform vec3 objectSizes[objectCount];
uniform float objectRoundings[objectCount];
uniform float objectSmoothings[objectCount];

// combined smoothing factors

uniform float jointEdgeSmoothing;
uniform float skelObjectSmoothing;

// scene settings
uniform mat4 sceneTransform;

/*
Affine Transformations
*/

// Translation matrix
mat4 translate(vec3 pos) 
{
    return mat4(
        vec4(1, 0, 0, 0),
        vec4(0, 1, 0, 0),
        vec4(0, 0, 1, 0),
		vec4(pos.x, pos.y, pos.z, 1)
    );
}

mat2 rot2D(float angle){
    float s = sin(angle);
    float c = cos(angle);
    return mat2(c, -s, s, c);
}

// Rotation matrix around the X axis.
mat4 rotateX(float theta) 
{
    float c = cos(theta);
    float s = sin(theta);
    return mat4(
        vec4(1, 0, 0, 0),
        vec4(0, c, -s, 0),
        vec4(0, s, c, 0),
		vec4(0, 0, 0, 1)
    );
}

// Rotation matrix around the Y axis.
mat4 rotateY(float theta) 
{
    float c = cos(theta);
    float s = sin(theta);
    return mat4(
        vec4(c, 0, s, 0),
        vec4(0, 1, 0, 0),
        vec4(-s, 0, c, 0),
		vec4(0, 0, 0, 1)
    );
}

// Rotation matrix around the Z axis.
mat4 rotateZ(float theta) 
{
    float c = cos(theta);
    float s = sin(theta);
    return mat4(
        vec4(c, -s, 0, 0),
        vec4(s, c, 0, 0),
        vec4(0, 0, 1, 0),
		vec4(0, 0, 0, 1)
    );
}

vec2 rotate2D(vec2 v, float a) 
{
	return vec2(cos(a)*v.x + sin(a)*v.y, -sin(a)*v.x + cos(a)*v.y);
}


// Rotation matrix around arbitrary axis
mat4 rotateAxis(vec3 axis, float angle)
{
	axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

vec3 rotatePt(vec3 pt, vec3 axis, float angle) {
	mat4 m = rotateAxis(axis, angle);
	return (m * vec4(pt, 1.0)).xyz;
}

/*
Constructive solid geometry
*/

// intersection
float intersectSDF(float distA, float distB) 
{
    return max(distA, distB);
}

// union
float unionSDF(float distA, float distB) 
{
    return min(distA, distB);
}

// difference
float differenceSDF(float distA, float distB) 
{
    return max(distA, -distB);
}

/*
Smoothing Operations
*/

// exponential smooth
float exp_smin( float a, float b, float k )
{
    float res = exp( -k*a ) + exp( -k*b );
    return -log( res )/k;
}

// polynomial smooth
float poly_smin( float a, float b, float k )
{
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

// power smooth
float power_smin( float a, float b, float k )
{
    a = pow( a, k ); b = pow( b, k );
    return pow( (a*b)/(a+b), 1.0/k );
}

// polynomial smooth min which returns both signed distance and interpolation factor
vec2 poly_smin_surface( float a, float b, float k )
{
    float h = max( k-abs(a-b), 0.0 )/k;
    float m = h*h*0.5;
    float s = m*k*(1.0/2.0);
    return (a<b) ? vec2(a-s,m) : vec2(b-s,1.0-m);
}

// surface smooth union
Surface union_surface(in Surface surface1, in Surface surface2, in float smoothness)
{
    vec2 sd = poly_smin_surface(surface1.signedDistance, surface2.signedDistance, smoothness);

    float signedDistance = sd[0];
    float interpol = sd[1];
    

    vec3 mixColor = mix(surface1.color, surface2.color, interpol);
    float mixAmbientScale = mix(surface1.ambientScale, surface2.ambientScale, interpol);
    float mixDiffuseScale = mix(surface1.diffuseScale, surface2.diffuseScale, interpol);
    float mixSpecularScale = mix(surface1.specularScale, surface2.specularScale, interpol);
    float mixSpecularPow = mix(surface1.specularPow, surface2.specularPow, interpol);
    float mixOcclusionScale = mix(surface1.occlusionScale, surface2.occlusionScale, interpol);
    float mixOcclusionRange = mix(surface1.occlusionRange, surface2.occlusionRange, interpol);
    float mixOcclusionResolution = mix(surface1.occlusionResolution, surface2.occlusionResolution, interpol);
    vec3 mixOcclusionColor = mix(surface1.occlusionColor, surface2.occlusionColor, interpol);

    return Surface(mixColor, mixAmbientScale, mixDiffuseScale, mixSpecularScale, mixSpecularPow, mixOcclusionScale, mixOcclusionRange, mixOcclusionResolution, mixOcclusionColor, signedDistance);
}

/*
Signed Distance Functions
*/

// Sphere (radius r)
float sphereSDF(vec3 p, float r) 
{
    return length(p) - r;
}

// Sphere Ripple
float rippleSphereSDF(vec3 pos, float radius, vec3 frequencies, vec3 amplitudes, vec3 phases)
{
    vec3 normPos = normalize(pos);

    vec3 axisAlign = normPos;

    vec3 surfaceDeformShape = vec3(cos(axisAlign.x * frequencies.x + phases.x), cos(axisAlign.y * frequencies.y + phases.y), cos(axisAlign.z * frequencies.z + phases.z));

    vec3 surfaceDeformScale = vec3(1.0 - abs(axisAlign.x), 1.0 - abs(axisAlign.y), 1.0 - abs(axisAlign.z)) * amplitudes;

    vec3 surfaceDeform = surfaceDeformShape * surfaceDeformScale;

    return length(pos) - radius + length(surfaceDeform);
}

// Box
float boxSDF( vec3 p, vec3 size )
{
    vec3 d = abs(p) - (size / 2.0);
    
    // Assuming p is inside the cube, how far is it from the surface?
    // Result will be negative or zero.
    float insideDistance = min(max(d.x, max(d.y, d.z)), 0.0);
    
    // Assuming p is outside the cube, how far is it from the surface?
    // Result will be positive or zero.
    float outsideDistance = length(max(d, 0.0));
    
    return insideDistance + outsideDistance;
}

// Box Round
float roundBoxSDF( vec3 p, vec3 size, float radius )
{
    vec3 d = abs(p) - ((size - radius) / 2.0);
    
    // Assuming p is inside the cube, how far is it from the surface?
    // Result will be negative or zero.
    float insideDistance = min(max(d.x, max(d.y, d.z)), 0.0);
    
    // Assuming p is outside the cube, how far is it from the surface?
    // Result will be positive or zero.
    float outsideDistance = length(max(d, 0.0));
    
    return insideDistance + outsideDistance - radius;
}

// Box Ripple
float rippleBoxSDF(vec3 pos, vec3 size, float radius, vec3 frequencies, vec3 amplitudes, vec3 phases)
{
    vec3 normPos = vec3(clamp(pos.x, -size.x / 2, size.x / 2) / size.x, clamp(pos.y, -size.y / 2, size.y / 2) / size.y, clamp(pos.z, -size.z / 2, size.z / 2) / size.z);

    vec3 surfaceDeformShape = vec3(0.0, 0.0, 0.0);
    surfaceDeformShape.y += cos(normPos.x * frequencies.x + phases.x);
    surfaceDeformShape.z += cos(normPos.x * frequencies.x + phases.x);
    surfaceDeformShape.x += cos(normPos.y * frequencies.y + phases.y);
    surfaceDeformShape.z += cos(normPos.y * frequencies.y + phases.y);
    surfaceDeformShape.x += cos(normPos.z * frequencies.z + phases.z);
    surfaceDeformShape.y += cos(normPos.z * frequencies.z + phases.z);

    vec3 surfaceDeform = surfaceDeformShape * amplitudes;

    vec3 d = abs(pos) - ((size + surfaceDeform - radius) / 2.0);
    
    // Assuming p is inside the cube, how far is it from the surface?
    // Result will be negative or zero.
    float insideDistance = min(max(d.x, max(d.y, d.z)), 0.0);
    
    // Assuming p is outside the cube, how far is it from the surface?
    // Result will be positive or zero.
    float outsideDistance = length(max(d, 0.0));
    
    return insideDistance + outsideDistance - radius;
}

// Cylinder(XY aligned, height h and radius r)
float cylinderSDF(vec3 p, float h, float r) 
{
    // How far inside or outside the cylinder the point is, radially
    float inOutRadius = length(p.xy) - r;
    
    // How far inside or outside the cylinder is, axially aligned with the cylinder
    float inOutHeight = abs(p.z) - h/2.0;
    
    // Assuming p is inside the cylinder, how far is it from the surface?
    // Result will be negative or zero.
    float insideDistance = min(max(inOutRadius, inOutHeight), 0.0);

    // Assuming p is outside the cylinder, how far is it from the surface?
    // Result will be positive or zero.
    float outsideDistance = length(max(vec2(inOutRadius, inOutHeight), 0.0));
    
    return insideDistance + outsideDistance;
}

// Cylinder(XY aligned, height h and radius r)
float roundCylinderSDF(vec3 p, float h, float r, float radius) 
{
    // How far inside or outside the cylinder the point is, radially
    float inOutRadius = length(p.xy) - (r - radius);
    
    // How far inside or outside the cylinder is, axially aligned with the cylinder
    float inOutHeight = abs(p.z) - (h - radius)/2.0;
    
    // Assuming p is inside the cylinder, how far is it from the surface?
    // Result will be negative or zero.
    float insideDistance = min(max(inOutRadius, inOutHeight), 0.0);

    // Assuming p is outside the cylinder, how far is it from the surface?
    // Result will be positive or zero.
    float outsideDistance = length(max(vec2(inOutRadius, inOutHeight), 0.0));
    
    return insideDistance + outsideDistance - radius;
}

// Cylinder Ripple
float rippleCylinderSDF(vec3 pos, float height, float r, float radius, vec3 frequencies, vec3 amplitudes, vec3 phases) 
{
    //vec3 normPos = vec3(clamp(pos.x, -radius, radius) / radius, clamp(pos.y, -radius, radius) / radius, clamp(pos.z, -height/2, height/2) / (height / 2));

    vec3 normPos = vec3(normalize(pos).x, normalize(pos).y, clamp(pos.z, -height/2, height/2) / (height / 2));

    vec3 surfaceDeformShape = vec3(0.0, 0.0, 0.0);

    surfaceDeformShape.x += cos(normPos.z * frequencies.z + phases.z);
    surfaceDeformShape.x += cos(normPos.y * frequencies.y + phases.y);
    surfaceDeformShape.y += cos(normPos.x * frequencies.x + phases.x);
    surfaceDeformShape.y += cos(normPos.z * frequencies.z + phases.z);
    surfaceDeformShape.z += cos(normPos.x * frequencies.x + phases.x);
    surfaceDeformShape.z += cos(normPos.y * frequencies.y + phases.y);

    vec3 surfaceDeformScale = amplitudes * 0.2;
    vec3 surfaceDeform = surfaceDeformShape * surfaceDeformScale;

    // How far inside or outside the cylinder the point is, radially
    float inOutRadius = length(pos.xy) - (r - radius) + length(surfaceDeform.xy);

    // How far inside or outside the cylinder is, axially aligned with the cylinder
    float inOutHeight = abs(pos.z) - (height/2.0 - radius)/2.0 + surfaceDeform.z;
    
    // Assuming p is inside the cylinder, how far is it from the surface?
    // Result will be negative or zero.
    float insideDistance = min(max(inOutRadius, inOutHeight), 0.0);

    // Assuming p is outside the cylinder, how far is it from the surface?
    // Result will be positive or zero.
    float outsideDistance = length(max(vec2(inOutRadius, inOutHeight), 0.0));
    
    return insideDistance + outsideDistance - radius;
}


// Capsule (XY aligned, height h and radius r)
float CapsuleSDF( vec3 p, float h, float r )
{
	p.z -= clamp( p.z, -h / 2.0, h / 2.0 );
	return length( p ) - r;
}

// Capsule (XY aligned, height h and radius r)
float roundCapsuleSDF( vec3 p, float h, float r, float radius )
{
	p.z -= clamp( p.z, -(h - radius) / 2.0, (h - radius) / 2.0 );
	return length( p ) - r - radius;
}


/*
Fractal Distance Functions
*/

float truchetTower (vec3 p, float r)
{

    float rnd = fract(sin(dot(floor(p) + 41., vec3(7.63, 157.31, 113.97)))*43758.5453);
    float qxy = 0.0;

    if (rnd>.75) p = 1. - p;
    else if(rnd>.5) p = p.yzx;
    else if(rnd>.25) p = p.zxy;

    p = fract(p); 

    // Draw three toroidal shapes within the unit block, oriented in such a way to form a 3D tile.
    // It can be a little frustrating trying to get the orientaion right, but once you get the hang
    // of it, it really is pretty simple. If you're not sure what's going on, have a look at the 
    // picture in the link provided above. By the way, the following differs a little from the
    // standard torii distance equations on account of slight mutations, cost cutting, etc, but 
    // that's what it all essentially amounts to.  
    
    // Toroidal shape one.
    vec3 q = p; // Not rotated.
    qxy = length(vec2(length(q.xy), q.z) - .5) + .175; // The "abs" and ".125" are additions, in this case.          ------REPLACE LENGTH BY ABS FOR SHARP VERSION
    q.xy = vec2(qxy, qxy);
    rnd = dot(q.xy, q.xy); // Reusing the "rnd" variable. Squared distance.

    // Toroidal shape two. Same as above, but rotated and shifted to a different part of the cube. 
    q = p.yzx - vec3(1, 1, 0); 
    qxy = length(vec2(length(q.xy), q.z) - .5) + .175;
    q.xy = vec2(qxy, qxy);
    
    rnd = min(rnd, dot(q.xy, q.xy)); // Minimum of shape one and two.
    
    // Toroidal shape three. Same as the two above, but rotated and shifted to a different part of the cube.
    q = p.zxy - vec3(0, 1, 0);
    qxy = length(vec2(length(q.xy), q.z) - .5) + .175;
    q.xy = vec2(qxy, qxy);
    rnd = min(rnd, dot(q.xy, q.xy)); // Minimum of of all three.
            
    return sqrt(rnd) - r; // Taking the square root and setting tube radius... kind of.

}

float apollonian1 (vec3 p, float aF1)
{
    float scale = 1;
    vec4 orb = vec4(1000,1000,1000,1000); 
    
    for( int i=0; i < 8; i++ )
    {
        p = -1.0 + 2.0*fract(0.5*p+0.5);
        float r2 = dot(p,p);
        orb = min( orb, vec4(abs(p),r2) );
        float k = aF1/ r2;
        p *= k;
        scale *= k;
    }
    float res = min(abs(p.z)+abs(p.x), 
            min(abs(p.x)+abs(p.y),
                abs(p.y)+abs(p.z)));
    
    return 0.25*res/scale;
}

vec4 apollonianColor(vec3 p)
{
    float scale = 1;
    vec4 orb = vec4(1000,1000,1000,1000); 
    float r2;
    for( int i=0; i < 10;i++ )
    {
        p = -1.0 + 2.0*fract(0.5*p+0.5);
        r2 = dot(p,p);
        orb = min( orb, vec4(abs(p),r2) );
        float k = 1.2/ r2;
        p *= k;
        scale *= k;
    }
    float res = min(abs(p.z)+abs(p.x), 
            max(abs(p.x)+abs(p.y),
                abs(p.y)+abs(p.z)));

    vec3 color = vec3 (r2,scale/100,0);
    return vec4(color, 0.25*res/scale);
}

float forks(vec3 p)
{
    float s = 0.1; 
    float scale = 1;

    float tt = 0.45+(sin(vectorTime[0]*2.)*.5+.5)*.65;
    
    tt = 0.5;
    
    vec4 orb = vec4(1000.0,1000.0,1000.0,1000.0); 
    
    for( int i=0; i<8;i++ )
    {
        p = -1.0 + 2.0*fract(0.5*p+0.5);

        float r2 = dot(p,p);
        
        
        r2 = r2 * tt + max(abs(p.x), max(abs(p.y),abs(p.z))) * (1.0 - tt);
        
        orb = min( orb, vec4(abs(p),r2) );
        
        float k = s/r2;
        p     *= k;
        scale *= k;
    }
    
    return 0.25*abs(p.y)/scale;
}

vec3 fold (vec3 p, vec3 n ) 
{
    return p-=2.0 * min(0.0, dot(p, n)) * n;
}

float tetraFold(vec3 p, float aF0, float aF1, float aF2, float aF3)
{
    float SCALE = 2;
    vec3 c;
    int n = 0;
    int IT = 8;
    float ic = 0;
    mat4 mat;
    vec3 Offset = vec3(1,1,1);

    aF0 = radians(aF0);
    aF1 = radians(aF1);
    aF2 = radians(aF2);
    aF3 = radians(aF3);

    while (n < IT) {


        if(p.x-p.y<0)p.xy = p.yx ;
        if(p.x-p.z<0)p.xz = p.zx ;
        if(p.y-p.z<0)p.yz = p.zy ;
        if(p.x+p.y<0)p.xy = -p.yx ;
        if(p.x+p.z<0)p.xz = -p.zx ;
        if(p.y+p.z<0)p.yz = -p.zy ;
        mat = rotateY( aF0 );
        p = (vec4(p, 1.0) * mat).xyz;
        p = p * SCALE - Offset * (SCALE-1);
        
        mat = rotateY(aF1 );
        p = (vec4(p, 1.0) * mat).xyz;
        n++;
    }

    float res= (length(p)-2) * pow(SCALE, -n);
    return  res;
}

float cubeFold(vec3 p, float aF0, float aF1, float aF2)
{
    float SCALE = aF2;
    vec3 c;
    int n = 0;
    int IT = 11;
    float ic = 0;
    mat4 mat;
    vec3 Offset = vec3(1,1,1);

    aF0 = radians(aF0);
    aF1 = radians(aF1);


    while (n < IT) {


        p.x = abs(p.x);

        p.y = abs(p.y);

        p.z = abs(p.z);

        mat = rotateY( aF1 );
        p = (vec4(p, 1.0) * mat).xyz;

        p = p * SCALE - Offset * (SCALE-1);

        mat = rotateY( aF0 );
        p = (vec4(p, 1.0) * mat).xyz;

        n++;
    }

    float res= (length(p)-2) * pow(SCALE, -n);
    return  res;
}

float cubeFold(vec3 p)
{
    float SCALE = 2.1;
    vec3 c;
    int n = 0;
    int IT = 10;
    float ic = 0;
    
    vec3 Offset = vec3(1,1,1);
    while (n < IT) {

        mat4 mat = rotateAxis(vec3 (0, 1,1), 2 * vectorTime[0]);
        p = (vec4(p, 1.0) * mat).xyz;
        
        if (p.x<0){
        p.x = abs(p.x);
             ic++;}
        if (p.y<0) 
        {
        p.y = abs(p.y); ic++;
        }
        if (p.z<0) {
        p.z = abs(p.z); ic++;
        }
       p = p * SCALE - Offset * (SCALE-1);

       n++;
    }

    float res= (length(p)-2) * pow(SCALE, -n);
    return  res;
}

float octaFold(vec3 p)
{
    float SCALE = 2.1;
    vec3 oldP = p;
    vec3 c;
    int n = 0;
    int IT = 10;

    mat4 mat = rotateAxis(vec3(-1,1,-1),-2*vectorTime[0]);
    p = (vec4(p, 1.0) * mat).xyz;

    vec3 Offset = vec3(1,1.5,1);
    while (n < IT) 
    {
        mat = rotateAxis(vec3(-1,1,-1),2*vectorTime[0]);
        p = (vec4(p, 1.0) * mat).xyz;

        p = abs(p);

        if(p.x-p.y<0) p.xy = p.yx; // fold 1

        if(p.x-p.z<0) p.xz = p.zx; // fold 2

        if(p.y-p.z<0) p.zy = p.yz; // fold 3       
        p = p * SCALE - Offset * (SCALE-1);
        mat = rotateAxis(vec3(1,-1,1), 1*vectorTime[0]);
        p = (vec4(p, 1.0) * mat).xyz;
        n++;
    }

    float res;
    res= (length(p)-2) * pow(SCALE, -n);
    
    return  res;
}

vec2 octaFold2(vec3 p )
{
    float SCALE = 3;
    vec3 c;
    int n = 0;
    int IT = 8;
    float val = 0;

    float t = 0.1* vectorTime[1];
    mat4 rot = rotateAxis(vec3 (0,1,0), -t *PI/4);
    p = (vec4(p, 1.0) * rot ).xyz;

    vec3 Offset = vec3(1,1,0.5);
    while (n < IT) 
    {
        
        mat4 mat = rotateAxis(vec3 (0,1,0), t* PI/4);

        p = (vec4(p, 1.0) * mat).xyz;

        if (p.x<0) { p.x = -p.x; val++;  }
        if (p.y<0) {p.y = -p.y;  val++;  }
        if (p.z<0) {p.z = -p.z ; val++;  }

        if(p.x-p.y<0){ p.xy = p.yx; val++;}
        if(p.x-p.z<0) {p.xz = p.zx; val++;}
        if(p.y-p.z<0) {p.zy = p.yz; val++;}    

        p.x = p.x * SCALE - Offset.x * (SCALE-1);
        p.y = p.y * SCALE - Offset.y * (SCALE-1);
        p.z = p.z * SCALE ;

        if(p.z>0.5*Offset.z*(SCALE-1)) p.z = p.z - Offset.z*(SCALE-1);

        mat = rotateAxis(vec3 (1,0,0), PI/2);
        p = (vec4(p, 1.0) * mat).xyz;
        n++;
    }
    
    val = val /(6*IT);

    float res = (length(p)-2) * pow(SCALE, -n);
    
    return  vec2(res, val);
}

float octaFold(vec3 p, float algoFloat0, float algoFloat1, float algoFloat2, float algoFloat3)
{
    float SCALE = 2.1;
    vec3 oldP = p;
    vec3 c;
    int n = 0;
    int IT = 10;

    mat4 mat = rotateAxis(vec3(-1,1,-1),-algoFloat0);
    p = (vec4(p, 1.0) * mat).xyz;

    vec3 Offset = vec3(1,1.5,1);
    while (n < IT) 
    {
        mat = rotateAxis(vec3(-1,1,-1), algoFloat0);
        p = (vec4(p, 1.0) * mat).xyz;

        p = abs(p);

        if(p.x-p.y<0) p.xy = p.yx; // fold 1

        if(p.x-p.z<0) p.xz = p.zx; // fold 2

        if(p.y-p.z<0) p.zy = p.yz; // fold 3       
        p = p * SCALE - Offset * (SCALE-1);
        mat = rotateAxis(vec3(1,-1,1), algoFloat1);
        p = (vec4(p, 1.0) * mat).xyz;
        n++;
    }

    float res;
    res= (length(p)-2) * pow(SCALE, -n);
    
    return  res;
}

float Thing2(vec3 p){
    int MI = 12;
    vec3 CSize = vec3( 0.92436,1.21212,1.0101);
    //Just scale=1 Julia box
    float DEfactor=1;
    vec3 ap = p+1;
    vec4 orbitTrap = vec4(0,0,0,0);
    float Size = 1.14664;
    vec3 C = vec3(0.28572,0.3238,-0.05716);
    vec3 Offset = vec3( 0.88888,0.4568,0.03704);
    float DEoffset = 0;
    if(!(ap.x==p.x && ap.y==p.y && ap.z==p.z))
    {
        for(int i=0 ; i<MI;i++)
        {
        ap=p;
        p=2.*clamp(p, -CSize, CSize)-p;
    
        float r2=dot(p,p);
        orbitTrap = min(orbitTrap, abs(vec4(p,r2)));
        float k=max(Size/r2,1.);

        p*=k;DEfactor*=k;
    
        p+=C;
        orbitTrap = min(orbitTrap, abs(vec4(p,dot(p,p))));
        }
    }
    //Call basic shape and scale its DE
    //return abs(0.5*Thingy(p,TThickness)/DEfactor-DEoffset);
    
    //Alternative shape
    //return abs(0.5*RoundBox(p, float3(1.,1.,1.), 1.0)/DEfactor-DEoffset);
    //Just a plane
    return abs(0.5*abs(p.z-Offset.z)/DEfactor-DEoffset);
}

float kaliBox(vec3 pos) {

    vec3 Trans =  vec3( 0.0365,-1.8613,0.0365);
    vec3 Julia = vec3(-0.6691,-1.3028,-0.45775);
    float scale =  2.04348;
    vec4 orbitTrap = vec4(0,0,0,0);        
    float MinRad2 = 0.3492;
    int Iterations = 15;
    float absScalem1 = abs(scale - 1.0);

    float AbsScaleRaisedTo1mIters = pow(abs(scale), float(1-Iterations));
        
    vec4 p = vec4(pos,1), p0 = vec4(Julia,1);  // p.w is the distance estimate
    
    for (int i=0; i<Iterations; i++) {

        p.xyz=abs(p.xyz)+Trans;
        float r2 = dot(p.xyz, p.xyz);
        if (i<2) orbitTrap = min(orbitTrap, abs(vec4(p.xyz,r2)));
        p *= clamp(max(MinRad2/r2, MinRad2), 0.0, 1.0);  // dp3,div,max.sat,mul
        p = p*scale + p0;
    
    }
    return ((length(p.xyz) - absScalem1) / p.w - AbsScaleRaisedTo1mIters);
}

float mandelbulb(vec3 p)
{
  vec3 c = p;
  float r = length(c);
  float dr = 1;
  float xr =0;
  float theta = 0;
  float phi = 0 ;

  for (int i = 0; i < 4 && r < 3; i++)
  {
    xr = pow(r, 7);
    dr = 6 * xr * dr + 1;
    theta = atan(c.y, c.x) * 8;
    phi = asin(clamp(c.z / r, -1,1)) * 8 - vectorTime.y;
    r = xr * r;
    c = r * vec3(cos(phi) * cos(theta), cos(phi) * sin(theta), sin(phi));
   
    c += p;
    r = length(c);
  }

  return 0.35 * log(r) * r / dr;
}

float mandelbulb_v2(vec3 p, vec3 scale)
{
  vec3 c = p;
  float r = length(c);
  float dr = 1;
  float xr =0;
  float theta = 0;
  float phi = 0 ;

  for (int i = 0; i < 4 && r < 3; i++)
  {
    xr = pow(r, 7);
    dr = 6 * xr * dr + 1;
    theta = atan(c.y, c.x) * (scale.x + 1.0) ;
    //phi = asin(clamp(c.z / r, -1,1)) * (scale.y + 1.0) - scale.z;
    phi = asin(clamp(c.z / r, -1,1)) * (scale.y + 1.0) - vectorTime.y / 10.0;
    r = xr * r;
    c = r * vec3(cos(phi) * cos(theta), cos(phi) * sin(theta), sin(phi));
   
    c += p;
    r = length(c);
  }

  return 0.35 * log(r) * r / dr;
}

float mandelbulbSDF( vec3 p, float power ) {
	vec3 w = p;
    float m = dot(w,w);
	float dz = 1.0;

	for( int i=0; i<3; i++ )
    {
        dz = power*pow(sqrt(m), power - 1.0 )*dz + 1.0;

        float r = length(w);
        float b = power*acos( w.y/r);
        float a = power*atan( w.x, w.z );
        w = p + pow(r,power) * vec3( sin(b)*sin(a), cos(b), sin(b)*cos(a) );

        m = dot(w,w);
		if( m > 256.0 )
            break;
    }

    return 0.25*log(m)*sqrt(m)/dz;
}

// DE: Infinitely tiled Menger IFS.
//
// For more info on KIFS, see:
// http://www.fractalforums.com/3d-fractal-generation/kaleidoscopic-%28escape-time-ifs%29/
float merger(in vec3 z)
{
    vec3 offset = vec3(0.92858,0.92858,0.32858);
    float scale = 3.0;
    const int iterations = 7;

	// Folding 'tiling' of 3D space;
	z  = abs(1.0-mod(z,2.0));

	float d = 1000.0;
	for (int n = 0; n < iterations; n++) {
		z.xy = rotate2D(z.xy,4.0+2.0*cos( vectorTime.y/8.0));		
		z = abs(z);
		if (z.x<z.y){ z.xy = z.yx;}
		if (z.x< z.z){ z.xz = z.zx;}
		if (z.y<z.z){ z.yz = z.zy;}
		z = scale*z-offset*(scale-1.0);
		if( z.z<-0.5*offset.z*(scale-1.0))  z.z+=offset.z*(scale-1.0);
		d = min(d, length(z) * pow(scale, float(-n)-1.0));
	}
	
	return d-0.001;
}

float sierpinskiPyramid(vec3 pt) {
    vec3 ori = vec3(0.0,2.5,0.0);
    vec3 a1 = vec3(1,1,1)+ori;
	vec3 a2 = vec3(-1,-1,1)+ori;
	vec3 a3 = vec3(1,-1,-1)+ori;
	vec3 a4 = vec3(-1,1,-1)+ori;
    
    vec3 c;
    int n = 0;
    float dist, d;
    float scale = 2.;
    while(n < 16) {
        c = a1;
        dist = length(pt - a1);
        d = length(pt - a2);
        if(d < dist) { 
            c = a2;
            dist = d;
        }
        d = length(pt - a3);
        if(d < dist) { 
            c = a3;
            dist = d;
        }
        d = length(pt - a4);
        if(d < dist) { 
            c = a4;
            dist = d;
        }
        pt = scale * pt - c * (scale - 1.0);
        n++;
    }
    
    return length(pt) * pow(scale, float(-n));
}

float sierpinskiPyramidFold(vec3 pt) {
    float r;
    float offset = 1.;
    float scale = 2.;
    pt.y -= 2.5;
    int n = 0;
    while(n < 15) {
        if(pt.x + pt.y < 0.) pt.xy = -pt.yx;
        if(pt.x + pt.z < 0.) pt.xz = -pt.zx;
        if(pt.y + pt.z < 0.) pt.zy = -pt.yz;
        pt = pt * scale - offset*(scale - 1.0);
        n++;
    }
    
    return (length(pt) * pow(scale, -float(n)));
}

float mengerSponge(vec3 pt) {
    float scale = 1.0;
    float offset = -2.;
    float iterations = 3.;

    float dist = boxSDF(vec3(pt.x, pt.y+offset, pt.z), vec3(scale));
    
    float s = 1.;
    
    float da, db, dc;
    
    for(int i = 0; i < 4; i++) {
        vec3 a = mod(pt * s, 2.0) - 1.0;
        s *= iterations;
        vec3 r = abs(1.0 - 3.0*abs(a));
        
        da = max(r.x, r.y);
        db = max(r.y, r.z);
        dc = max(r.z, r.x);
        
        float c = (min(da, min(db, dc)) - 1.) / s;
        if ( c > dist) dist = c;
    }
    
    return dist;
}

float alteredMenger(vec3 pt){
    float scale = 1.0;
    float offset = -2.;
    float iterations = 3.;

    float dist = boxSDF(vec3(pt.x, pt.y+offset, pt.z), vec3(scale));
    
    float s = 2.;
    
    float da, db, dc;
    
    for(int i = 0; i < 4; i++) {
        vec3 a = mod(pt * s, 2.0) - 1.0;
        s *= iterations;
        vec3 r = abs(1.0 - 3.0*abs(a));
        
        r = (vec4(r, 1.0) * rotateX(20.)).xyz;
        da = max(r.x+1.5, r.y);
        r = (vec4(r, 1.0) * rotateX(80.)).xyz;
        
        da = max(da + r.x-0.5, r.y);
        db = max(r.y, r.z);
        dc = max(r.z+0.5, r.x);
        
        float c = (min(da, min(db, dc)) - 1.) / s;
        if ( c > dist) dist = c;
    }
    
    return dist;
}

float evolvingFractal(vec3 pt) {
    //vec3 off = vec3(0.7, 0.5, 0.15);
    //vec3 off = vec3(0.5, 0.85, 1.25);
    vec3 off = vec3(1.);
    float scale = 2.0;
    pt.y -= 2.5;
    int n = 0;

    while(n < 18) {
        //pt = rotatePt(pt, vec3(1.), 31.); // snowflake
        pt = rotatePt(pt, vec3(1), cos(10.+vectorTime.y*0.1));
        
        pt = abs(pt); // for cube
        
        if(pt.x + pt.y < 0.) pt.xy = -pt.yx;
        if(pt.x + pt.z < 0.) pt.xz = -pt.zx;
        if(pt.y + pt.z < 0.) pt.zy = -pt.yz;
          
        pt = rotatePt(pt, vec3(0.35,0.2,0.3), -90.+vectorTime.y*0.1);
        
        pt.x = pt.x * scale - off.x*(scale - 1.0);
        pt.y = pt.y * scale - off.y*(scale - 1.0);
        pt.z = pt.z * scale - off.z*(scale - 1.0);
        
        pt = rotatePt(pt, vec3(0.3,0.1,0.25), -70.+vectorTime.y*0.1);
        
        n++;
    }
    
    return (length(pt) * pow(scale, -float(n)));
}

float evolvingFractal2(vec3 pt) {
    vec3 off = vec3(1.25);
    float scale = 2.;
    pt.y -= 2.5;
    int n = 0;

    while(n < 18) {
        //pt = rotatePt(pt, vec3(1.), 31.); // snowflake
        pt = rotatePt(pt, vec3(1.), sin(0.+vectorTime.y*0.1));
        
        pt = abs(pt); // for cube
        
        if(pt.x + pt.y < 0.) pt.xy = -pt.yx;
        if(pt.x + pt.z < 0.) pt.xz = -pt.zx;
        if(pt.y + pt.z < 0.) pt.zy = -pt.yz;
          
        pt = rotatePt(pt, vec3(0.35,0.2,0.3), -90.+vectorTime.y*0.1);
        
        pt.x = pt.x * scale - off.x*(scale - 1.0);
        pt.y = pt.y * scale - off.y*(scale - 1.0);
        pt.z = pt.z * scale - off.z*(scale - 1.0);
        
        pt = rotatePt(pt, vec3(0.3,0.1,0.25), -70.+vectorTime.y*0.1);
        
        n++;
    }
    return (length(pt) * pow(scale, -float(n)));
}


float julia(vec3 pos, vec3 scale)
{
    int iter_count = int(scale.y);
    float delta = scale.z;


    //float delta = sin(vectorTime.y / 2.0) / 4.0;
    vec4 c = vec4(-0.2, -.6, delta, 0.0);
    vec4 z = vec4(pos, 0.0);
    vec4 nz;

    float dr2   = 1.0;
    float r2    = dot(z,z);

    for(int i = 0; i < iter_count; i++)
    {
        /*
        dr2    *= 4.0 * r2;
        nz.x    = z.x * z.x - dot(z.yzw, z.yzw);
        nz.yzw  = 2.0 * z.x * z.yzw;
        */
        
        dr2    *= scale.x * r2;
        nz.x    = z.x * z.x - dot(z.yzw, z.yzw);
        nz.yzw  = scale.x / 2.0 * z.x * z.yzw;

        z = nz + c;

        r2 = dot(z, z);
        if(r2 > 4.0) {
            break; 
        }
    }

    return 0.25 * sqrt(r2 / dr2) * log(r2);
}

float snake(vec3 pos, vec3 scale, float radius)
{
    pos.z += vectorTime.y * .4;

    //repetition
    pos.xy = (fract(pos.xy) - .5); 
    pos.z = mod(pos.z, .25) - .125; 

    float box = roundBoxSDF(pos, vec3(scale.x/100, scale.y/100 , scale.z/100), radius);

    return box;
}

float snake_rippling(vec3 pos, vec3 scale, float radius)
{
    pos.z += vectorTime.y * .4;

    //repetition
    pos.xy = (fract(pos.xy) - .5); 
    //pos.z = mod(pos.z, .25) - .125; 
    pos.z = mod(pos.z, .5) - .25; 

    vec3 frequencies = objectFrequencies[0];
    vec3 amplitudes = objectAmplitudes[0];
    vec3 phases = objectPhases[0];

    float box = rippleBoxSDF(pos, vec3(scale.x/100, scale.y/100 , scale.z/100), radius, frequencies, amplitudes, phases);

    return box;
}

/*
Primitive Morphing
*/

float primitiveMorphSDF(vec3 p, vec3 size, float rounding, float primitive) 
{
    int primitive1Index = int(primitive);
    float primitiveMix = fract(primitive);
    int primitive2Index = primitive1Index;
    
    if(primitiveMix != 0.0) primitive2Index += 1;
    
    float primitive1dist = 0.0;
    float primitive2dist = 0.0;
    
    if (primitive1Index == 0) // sphere
    {
        primitive1dist = sphereSDF(p, size.x);    
        
        if (primitive2Index == primitive1Index + 1)
        {
            primitive2dist = roundBoxSDF(p, size, rounding);
        }
        else
        {
            primitive2dist = primitive1dist;
        }
    }
    else if(primitive1Index == 1) // box
    {
        primitive1dist = roundBoxSDF(p, size, rounding);  
        
        if (primitive2Index == primitive1Index + 1)
        {
            primitive2dist = roundCapsuleSDF(p, size.z, size.x, rounding);
        }
        else
        {
            primitive2dist = primitive1dist;
        }
    }
    else if(primitive1Index == 2) // capsule
    {
        primitive1dist = roundCapsuleSDF(p, size.z, size.x, rounding);
        
        if (primitive2Index == primitive1Index + 1)
        {
            primitive2dist = roundCylinderSDF(p, size.z, size.x, rounding);
        }
        else
        {
            primitive2dist = primitive1dist;
        }
    }
    else if(primitive1Index == 3) // Cylinder
    {
         primitive1dist = roundCylinderSDF(p, size.z, size.x, rounding);
         primitive2dist = primitive1dist;
    }
    else if(primitive1Index == 4) // truchetTower
    {
        primitive1dist = truchetTower(p, size.z); 
        primitive2dist = primitive1dist;
    }
    else if(primitive1Index == 5) // apollonian1
    {
        primitive1dist = apollonian1(p, size.z); 
        primitive2dist = primitive1dist;
    }
    else if(primitive1Index == 6) // Thing2
    {
        primitive1dist = Thing2(p); 
        primitive2dist = primitive1dist;
    }
    else if(primitive1Index == 7) // kaliBox
    {
        primitive1dist = kaliBox(p); 
        primitive2dist = primitive1dist;
    }        
    else if(primitive1Index == 8) // mandelbulb_v2
    {
        primitive1dist = mandelbulb_v2(p / size.z, size) * size.z; 
        primitive2dist = primitive1dist;
    }                
    else if(primitive1Index == 9) // merger
    {
        primitive1dist =  merger(p); 
        primitive2dist = primitive1dist;
    }    
    else if(primitive1Index == 10) // sierpinskiPyramid
    {
        primitive1dist = sierpinskiPyramid(p); 
        primitive2dist = primitive1dist;
    }                
    else if(primitive1Index == 11) // mengerSponge
    {
        primitive1dist = mengerSponge(p); 
        primitive2dist = primitive1dist;
    }   
    else if(primitive1Index == 12) // alteredMenger
    {
        primitive1dist = alteredMenger(p); 
        primitive2dist = primitive1dist;
    }                 
    else if(primitive1Index == 13) // evolvingFractal
    {
        primitive1dist = evolvingFractal(p); 
        primitive2dist = primitive1dist;
    }  
    else if(primitive1Index == 14) // evolvingFractal2
    {
        primitive1dist = evolvingFractal2(p); 
        primitive2dist = primitive1dist;
    }      
    else if(primitive1Index == 15) // mandelbulbSDF
    {
        primitive1dist = mandelbulbSDF(p, size.x); 
        primitive2dist = primitive1dist;
    }
    else if(primitive1Index == 16) // julia
    {
        primitive1dist = julia(p, size); 
        primitive2dist = primitive1dist;
    }    
    else if(primitive1Index == 17) // snake
    {
        //primitive1dist = snake(p, size, rounding); 

        primitive1dist = snake_rippling(p, size, rounding); 
        primitive2dist = primitive1dist;
    }    
        
    return mix(primitive1dist, primitive2dist, primitiveMix);

}

float primitiveMorphRippleSDF(vec3 p, vec3 size, float rounding, float primitive, vec3 frequency, vec3 amplitude, vec3 phase) 
{
    int primitive1Index = int(primitive);
    float primitiveMix = fract(primitive);
    int primitive2Index = primitive1Index;
    
    if(primitiveMix != 0.0) primitive2Index += 1;
    
    float primitive1dist = 0.0;
    float primitive2dist = 0.0;
    
    if (primitive1Index == 0) // sphere
    {
    
        if(amplitude == 0) // non-rippling
        {
            primitive1dist = sphereSDF(p, size.x);    
        }
        else
        {
            primitive1dist = rippleSphereSDF(p, size.x, frequency, amplitude, phase);
        }
        
        if (primitive2Index == primitive1Index + 1)
        {
            if(amplitude == 0) // non-rippling
            {
                primitive2dist = roundBoxSDF(p, size, rounding);
            }
            else
            {
                primitive2dist = rippleBoxSDF(p, size, rounding, frequency, amplitude, phase);
            }
        }
        else
        {
            primitive2dist = primitive1dist;
        }
    }
    else if(primitive1Index == 1) // box
    {
        if(amplitude == 0) // non-rippling
        {
            primitive1dist = roundBoxSDF(p, size, rounding);  
        }
        else
        {
            primitive1dist = rippleBoxSDF(p, size, rounding, frequency, amplitude, phase);
        }
        
        if (primitive2Index == primitive1Index + 1)
        {
            primitive2dist = roundCapsuleSDF(p, size.z, size.x, rounding);
        }
        else
        {
            primitive2dist = primitive1dist;
        }
    }
    else if(primitive1Index == 2) // capsule
    {
        primitive1dist = roundCapsuleSDF(p, size.z, size.x, rounding);
        
        if (primitive2Index == primitive1Index + 1)
        {
            if(amplitude == 0) // non-rippling
            {
                primitive2dist = roundCylinderSDF(p, size.z, size.x, rounding);
            }   
            else
            {
                primitive2dist = rippleCylinderSDF(p, size.z, size.x, rounding, frequency, amplitude, phase);
            }
        }
        else
        {
            primitive2dist = primitive1dist;
        }
    }
    else if(primitive1Index == 3) // Cylinder
    {
        if(amplitude == 0) // non-rippling
        {
            primitive1dist = roundCylinderSDF(p, size.z, size.x, rounding);
        }
        else
        {
            primitive1dist = rippleCylinderSDF(p, size.z, size.x, rounding, frequency, amplitude, phase);
        }
         
        primitive2dist = primitive1dist;
    }
    else if(primitive1Index == 4) // truchetTower
    {
        primitive1dist = truchetTower(p, size.z); 
        primitive2dist = primitive1dist;
    }
    else if(primitive1Index == 5) // apollonian1
    {
        primitive1dist = apollonian1(p, size.z); 
        primitive2dist = primitive1dist;
    }
    else if(primitive1Index == 6) // Thing2
    {
        primitive1dist = Thing2(p); 
        primitive2dist = primitive1dist;
    }
    else if(primitive1Index == 7) // kaliBox
    {
        primitive1dist = kaliBox(p); 
        primitive2dist = primitive1dist;
    }        
    else if(primitive1Index == 8) // mandelbulb_v2
    {
        primitive1dist = mandelbulb_v2(p / size.z, size) * size.z; 
        primitive2dist = primitive1dist;
    }                
    else if(primitive1Index == 9) // merger
    {
        primitive1dist =  merger(p); 
        primitive2dist = primitive1dist;
    }    
    else if(primitive1Index == 10) // sierpinskiPyramid
    {
        primitive1dist = sierpinskiPyramid(p); 
        primitive2dist = primitive1dist;
    }                
    else if(primitive1Index == 11) // mengerSponge
    {
        primitive1dist = mengerSponge(p); 
        primitive2dist = primitive1dist;
    }   
    else if(primitive1Index == 12) // alteredMenger
    {
        primitive1dist = alteredMenger(p); 
        primitive2dist = primitive1dist;
    }                 
    else if(primitive1Index == 13) // evolvingFractal
    {
        primitive1dist = evolvingFractal(p); 
        primitive2dist = primitive1dist;
    }  
    else if(primitive1Index == 14) // evolvingFractal2
    {
        primitive1dist = evolvingFractal2(p); 
        primitive2dist = primitive1dist;
    }      
    else if(primitive1Index == 15) // mandelbulbSDF
    {
        primitive1dist = mandelbulbSDF(p, size.x); 
        primitive2dist = primitive1dist;
    }
    else if(primitive1Index == 16) // julia
    {
        primitive1dist = julia(p, size); 
        primitive2dist = primitive1dist;
    }    
    else if(primitive1Index == 17) // snake
    {
        //primitive1dist = snake(p, size, rounding); 
        primitive1dist = snake_rippling(p, size, rounding); 
        primitive2dist = primitive1dist;
    }    
        
    return mix(primitive1dist, primitive2dist, primitiveMix);

}


/**
 * Signed distance function describing the scene.
 * 
 * Absolute value of the return value indicates the distance to the surface.
 * Sign indicates whether the point is inside or outside the surface,
 * negative indicating inside.
 */
float sceneSDF(vec3 samplePoint) 
{    
    vec4 samplePoint4D = sceneTransform * vec4(samplePoint, 1.0);
    
    float distJoints = 1000.0;
    
    for(int jI=0; jI<jointCount; ++jI)
    {
        if(jointPrimitives[jI] < 0) // do nothing
        {}
        else
        {
            distJoints = poly_smin( distJoints, primitiveMorphSDF((jointTransforms[jI] * samplePoint4D).xyz, jointSizes[jI], jointRoundings[jI], jointPrimitives[jI]), jointSmoothings[jI] );
        }
    }
    
    float distEdges = 1000.0;
    

    for(int eI=0; eI<edgeCount; ++eI)
    {
        if(edgePrimitives[eI] < 0) // do nothing
        {}
        else
        {
            distEdges = poly_smin( distEdges, primitiveMorphSDF((edgeTransforms[eI] * samplePoint4D).xyz, vec3(edgeSizes[eI].x, edgeSizes[eI].y, edgeLengths[eI] * edgeSizes[eI].z), edgeRoundings[eI], edgePrimitives[eI]), edgeSmoothings[eI] );
        }
    }
    
    float distObjects = 1000.0;
    
    for(int oI=0; oI<objectCount; ++oI)
    {    
        if(objectPrimitives[oI] < 0) // do nothing
        {}
        else
        {
            distObjects = poly_smin( distObjects, primitiveMorphRippleSDF((objectTransforms[oI] * samplePoint4D).xyz, objectSizes[oI], objectRoundings[oI], objectPrimitives[oI], objectFrequencies[oI], objectAmplitudes[oI], objectPhases[oI] ), objectSmoothings[oI] );
        }
    }
    
    float distJointEdges = 1000.0;
    
    if(distJoints < 1000.0 && distEdges < 1000.0)
    {
        distJointEdges = poly_smin(distJoints, distEdges, jointEdgeSmoothing);
    }
    else if(distJoints < 1000.0)
    {
        distJointEdges = distJoints;
    }
    else if(distEdges < 1000.0)
    {
        distJointEdges = distEdges;
    }
    
    float dist;
    
    if(distObjects < 1000.0)
    {
        dist = poly_smin(distJointEdges, distObjects, skelObjectSmoothing);
    }
    else
    {
        dist = distJointEdges;
    }
 
    return dist;
}

Surface sceneSDF_surface(vec3 samplePoint) 
{    
    vec4 samplePoint4D = sceneTransform * vec4(samplePoint, 1.0);

    // skeleton joints
    float distJoints = 1000.0;
    
    for(int jI=0; jI<jointCount; ++jI)
    {
        if(jointPrimitives[jI] < 0) // do nothing
        {}
        else
        {
            distJoints = poly_smin( distJoints, primitiveMorphSDF((jointTransforms[jI] * samplePoint4D).xyz, jointSizes[jI], jointRoundings[jI], jointPrimitives[jI]), jointSmoothings[jI] );
        }
    }
    
    Surface jointSurface = Surface(jointColor, jointAmbientScale, jointDiffuseScale, jointSpecularScale, jointSpecularPow, jointOcclusionScale, jointOcclusionRange, jointOcclusionResolution, jointOcclusionColor, distJoints);
    
    // skeleton edges
    float distEdges = 1000.0;
    
    for(int eI=0; eI<edgeCount; ++eI)
    {
        if(edgePrimitives[eI] < 0) // do nothing
        {}
        else
        {
            distEdges = poly_smin( distEdges, primitiveMorphSDF((edgeTransforms[eI] * samplePoint4D).xyz, vec3(edgeSizes[eI].x, edgeSizes[eI].y, edgeLengths[eI] * edgeSizes[eI].z), edgeRoundings[eI], edgePrimitives[eI]), edgeSmoothings[eI] );
        }
    }
    
    Surface edgeSurface = Surface(edgeColor, edgeAmbientScale, edgeDiffuseScale, edgeSpecularScale, edgeSpecularPow, edgeOcclusionScale, edgeOcclusionRange, edgeOcclusionResolution, edgeOcclusionColor, distEdges);

    // objects
    float distObjects = 1000.0;
    float maxDistObjects = 1000.0;
    Surface objectSurface = Surface(vec3(0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 10.0, 0.0, 0.5, 0.5, vec3(0.0, 0.0, 0.0), 1000.0);
    
    for(int oI=0; oI<objectCount; ++oI)
    {    
        distObjects = 1000.0;
    
        if(objectPrimitives[oI] < 0) // do nothing
        {}
        else
        {
            distObjects = poly_smin( distObjects, primitiveMorphRippleSDF((objectTransforms[oI] * samplePoint4D).xyz, objectSizes[oI], objectRoundings[oI], objectPrimitives[oI], objectFrequencies[oI], objectAmplitudes[oI], objectPhases[oI] ), objectSmoothings[oI] );
        }
        
        Surface tmpSurface = Surface(objectColors[oI], objectAmbientScales[oI], objectDiffuseScales[oI], objectSpecularScales[oI], objectSpecularPows[oI], objectOcclusionScales[oI], objectOcclusionRanges[oI], objectOcclusionResolutions[oI], objectOcclusionColors[oI], distObjects);
        objectSurface = union_surface(tmpSurface, objectSurface, objectSmoothings[oI]);
        
        if(maxDistObjects > distObjects)
        {
            maxDistObjects = distObjects;
        }
    }


    // combined surface from skeleton egdes and joints
    Surface skelEdgeSurface = Surface(vec3(0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 10.0, 0.0, 0.5, 0.5, vec3(0.0, 0.0, 0.0), 1000.0);
    
    if(distJoints < 1000.0 && distEdges < 1000.0)
    {
        skelEdgeSurface = union_surface(jointSurface, edgeSurface, jointEdgeSmoothing);
    }
    else if(distJoints < 1000.0)
    {
        skelEdgeSurface = jointSurface;
    }
    else if(distEdges < 1000.0)
    {
        skelEdgeSurface = edgeSurface;
    }
    
    Surface closestSurface;
    
    if(maxDistObjects < 1000.0)
    {
        closestSurface = union_surface(skelEdgeSurface, objectSurface, skelObjectSmoothing);
    }
    else
    {
        closestSurface = skelEdgeSurface;
    }

    return closestSurface;
}

float ambientOcclusion(vec3 surfacePos, vec3 surfaceNormal, float occlusionRange, float occlusionResolution)
{
    
    float minT = 0.01;
    float maxT = occlusionRange;
    float tIncr = occlusionResolution;
    
    float occlusionFacor = 1.0;
    float dist;
    
    for(float t = minT; t < maxT; t += tIncr)
    {
        dist = sceneSDF( surfacePos + surfaceNormal * t );
        
        if(dist < t - EPSILON)
        {
            float normT = (t - minT) / (maxT - minT);
            occlusionFacor = occlusionFacor * normT + occlusionFacor * dist / t * (1.0 - normT);
        }
        
        if(occlusionFacor <= 0.0) break;
    }
    
    return occlusionFacor;
}

float doAoSSS(vec3 p, vec3 n, float steps, float delta)
{
    float a = 0.0;
    float weight = .5;
    for (float i = 1; i <= steps; i += 1) 
    {
        float d = (i / steps) * delta;
        a = a + weight * (d - sceneSDF(p + n * d));
        weight = weight * 0.6;
    }
    
    return clamp(1.0 - a, 0.0, 1.0);
}

const int steps = 4;

float doAoSSS2(vec3 p, vec3 n, float delta)
{
    float a = 0.0;
    float weight = .5;
    for (int i = 1; i <= steps; i += 1) 
    {
        float d = (i / steps) * delta;
        a = a + weight * (d - sceneSDF(p + n * d));
        weight = weight * 0.6;
    }
    
    return clamp(1.0 - a, 0.0, 1.0);
}


/**
 * Return the shortest distance from the eyepoint to the scene surface along
 * the marching direction. If no part of the surface is found between start and end,
 * return end.
 * 
 * eye: the eye point, acting as the origin of the ray
 * marchingDirection: the normalized direction to march in
 * start: the starting distance away from the eye
 * end: the max distance away from the ey to march before giving up
 */
float shortestDistanceToSurface(vec3 eye, vec3 marchingDirection, float start, float end) 
{
    float depth = start;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) 
    {
        vec3 p = eye + depth * marchingDirection ;

        if (rayRotation != 0.0){
            p.xy *= rot2D(depth *.2 * (rayRotation));
        }

        if (rayWiggle != 0.0){
            // p.y += sin(depth*(rayWiggle + 1)*.5)*.35;
            //p.y += sin(depth*(rayWiggle)*.5 + vectorTime.y*.4)*.35 ;
            p.y += rayWiggle * sin(depth*(rayWiggle)*.5 + vectorTime.y*.4)*.35 ;
        }    

        float dist = sceneSDF(p);
        if (dist < EPSILON) 
        {
			return depth;
        }
        depth += dist;
        if (depth >= end) 
        {
            return end;
        }
    }
    return end;
}

Surface shortestDistanceToSurface_surface(vec3 eye, vec3 marchingDirection, float start, float end) 
{
    float depth = start;
    Surface closestSurface;

    for (int i = 0; i < MAX_MARCHING_STEPS; i++) 
    {
        vec3 p = eye + depth * marchingDirection ;
       
        if (rayRotation != 0.0){
            p.xy *= rot2D(depth *.2 * (rayRotation));
        }

        if (rayWiggle != 0.0){
            p.y += rayWiggle * sin(depth*(rayWiggle)*.5 + vectorTime.y*.4)*.35;
        }    
        

        closestSurface = sceneSDF_surface(p);

        if (closestSurface.signedDistance < EPSILON) 
        {
            closestSurface.signedDistance = depth;
            return closestSurface;
        }

        depth += closestSurface.signedDistance;

        if (depth >= end) 
        {
            closestSurface.signedDistance = end;
            closestSurface.color = bgColor;
            return closestSurface;
        }
    }
    
    closestSurface.color = bgColor;
    closestSurface.ambientScale = 1.0;
    closestSurface.diffuseScale = 0.0;
    closestSurface.specularScale = 0.0;
    closestSurface.specularPow = 1.0;
    closestSurface.occlusionScale = 0.0;
    closestSurface.occlusionRange = 1.0;
    closestSurface.occlusionResolution = 1.0;
    closestSurface.occlusionColor = bgOcclusionColor;
    closestSurface.signedDistance = end;
    
    return closestSurface;
}
            

/**
 * Return the normalized direction to march in from the eye point for a single pixel.
 * 
 * fieldOfView: vertical field of view in degrees
 * size: resolution of the output image
 * fragCoord: the x,y coordinate of the pixel in the output image
 */
vec3 rayDirection(float fieldOfView, vec2 fragCoord) 
{
    float z = 1.0 / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(fragCoord, -z));
}

/**
 * Using the gradient of the SDF, estimate the normal on the surface at point p.
 */
vec3 estimateNormal(vec3 p) 
{
    /*
    return normalize(vec3(
        sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)),
        sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)),
        sceneSDF(vec3(p.x, p.y, p.z  + EPSILON)) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON))
    ));
    */
    
    float center_dist = sceneSDF(vec3(p.x, p.y, p.z));

    return normalize(vec3(
        sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - center_dist,
        sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - center_dist,
        sceneSDF(vec3(p.x, p.y, p.z  + EPSILON)) - center_dist
    ));
    
    //return vec3(1.0, 0.0, 0.0);
}

/**
 * Lighting contribution of a single point light source via Phong illumination.
 * 
 * The vec3 returned is the RGB color of the light's contribution.
 *
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 * lightPos: the position of the light
 * lightIntensity: color/intensity of the light
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
vec3 phongContribForLight(vec3 p, vec3 eye, vec3 lightPos, vec3 N, vec3 diffuseColor, float diffuseScale, vec3 specularColor, float specularScale, float specularPow) 
{
    //vec3 N = estimateNormal(p);
    vec3 L = normalize(lightPos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));
    
    float dotLN = dot(L, N);
    float dotRV = dot(R, V);
    
    if (dotLN < 0.0) {
        // Light not visible from this point on the surface
        return vec3(0.0, 0.0, 0.0);
    } 
    
    if (dotRV < 0.0) {
        // Light reflection in opposite direction as viewer, apply only diffuse
        // component
        return diffuseScale * (diffuseColor * dotLN);
    }
    return diffuseScale * diffuseColor * dotLN + specularScale * specularColor * pow(dotRV, specularPow);
}

float hardShadow( vec3 lightPoint, vec3 lightToSurfaceDir, float mint, float maxt )
{
    for( float t=mint; t < maxt; )
    {
        float h = sceneSDF(lightPoint + lightToSurfaceDir*t);
        if( h<0.002 )
            return 0.0;
        t += h;
    }
    return 1.0;
}

float softShadow( vec3 lightPoint, vec3 lightToSurfaceDir, float mint, float maxt, float k )
{
    float res = 1.0;

    for( float t=mint; t < maxt; )
    {
        float h = sceneSDF(lightPoint + lightToSurfaceDir*t);
        if( h<0.002 )
            return 0.0;

        res = min( res, k*h/t );

        //t += clamp( h, 0.02, 0.04);

        t += h / 10.0;
    }
    return res;
}

/**
 * Return a transform matrix that will transform a ray from view space
 * to world coordinates, given the eye point, the camera target, and an up vector.
 *
 * This assumes that the center of the camera is aligned with the negative z axis in
 * view space when calculating the ray marching direction. See rayDirection.
 */
mat4 viewMatrix(vec3 eye, vec3 center, vec3 up) 
{
    // Based on gluLookAt man page
    vec3 f = normalize(center - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);
    return mat4(
        vec4(s, 0.0),
        vec4(u, 0.0),
        vec4(-f, 0.0),
        vec4(0.0, 0.0, 0.0, 1)
    );
}


void main()
{
    vec3 viewDir = rayDirection(camAngle, fragCoord);
    vec3 eye = camPosition;
    mat4 viewToWorld = viewMatrix(eye, vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, -1.0));
    vec3 worldDir = (viewToWorld * vec4(viewDir, 0.0)).xyz;
    
    Surface surface = shortestDistanceToSurface_surface(eye, worldDir, MIN_DIST, MAX_DIST);
    float dist = surface.signedDistance;
    
    if (dist > MAX_DIST - EPSILON) 
    {
        // Didn't hit anything
        fragColor = vec4(surface.color, 1.0);
        return;
    }
    
    // The closest point on the surface to the eyepoint along the view ray
    // used for lighting calculations
    vec3 surfacePos = eye + dist * worldDir;
    
    // begin test adding wiggle here
    if (rayRotation != 0.0){
        surfacePos.xy *= rot2D(dist *.2 * (rayRotation));
    }
    if (rayWiggle != 0.0){
        surfacePos.y += rayWiggle * sin(dist*(rayWiggle)*.5 + vectorTime.y*.4)*.35;
    }    
    // end test adding wiggle here
    
    // surface normal estimation
    vec3 surfaceNormal = estimateNormal(surfacePos);

    // regular lighting    
    vec3 color1 = surface.color * surface.ambientScale;
    
    // soft shadows
    float lightStrength = 1.0;
    
    if(shadowStrength > 0.0)
    {
        vec3 lightToSurfaceVec = surfacePos - lightPosition;
        float lightToSurfaceDist = length(lightToSurfaceVec);
        vec3 lightToSurfaceDir = normalize(lightToSurfaceVec);
        //lightStrength = hardShadow(lightPosition, lightToSurfaceDir, 0.0, lightToSurfaceDist - 0.01);
        lightStrength = softShadow(lightPosition, lightToSurfaceDir, 0.0, lightToSurfaceDist- 1.0, shadowSmooth);
        lightStrength = (1.0 - shadowStrength) + lightStrength * shadowStrength;
    }

    color1 += phongContribForLight(surfacePos, eye, lightPosition, surfaceNormal, surface.color, surface.diffuseScale, surface.color, surface.specularScale, surface.specularPow ) * lightStrength;
    
    // ambient occlusion
    vec3 colorDiff = color1 - surface.occlusionColor;
    float occlusionStrength = ambientOcclusion(surfacePos, surfaceNormal, surface.occlusionRange, surface.occlusionResolution);
    //float occlusionStrength = doAoSSS(surfacePos, surfaceNormal, surface.occlusionRange, surface.occlusionResolution);
    //float occlusionStrength = doAoSSS2(surfacePos, surfaceNormal, surface.occlusionResolution);
    //float occlusionStrength = 0.5;
    
   	occlusionStrength = 1.0 - occlusionStrength;
   	occlusionStrength *= surface.occlusionScale;
   	
   	vec3 color2 = vec3(color1.r - occlusionStrength * colorDiff.r, color1.g - occlusionStrength * colorDiff.g , color1.b - occlusionStrength * colorDiff.b);
   	
   	// distance fog
   	float fog_dist = clamp((dist - fog_min_dist) / (fog_max_dist - fog_min_dist), 0.0, 1.0);
   	
   	vec3 color3 = fog_dist * bgColor + (1.0 - fog_dist) * color2;
   	
    fragColor = vec4(color3, 1.0);
}