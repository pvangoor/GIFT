#pragma once
/*
  allow for selection of single versus double precision
 */
#if defined(USE_SINGLE_FLOAT)

// use single precision
#define ftype float
#define Vector2T Vector2f
#define Vector3T Vector3f
#define Vector4T Vector4f
#define Matrix3T Matrix3f
#define Matrix4T Matrix4f
#define MatrixXT MatrixXf

#else

// use double precision
#define ftype double
#define Vector2T Vector2d
#define Vector3T Vector3d
#define Vector4T Vector4d
#define Matrix3T Matrix3d
#define Matrix4T Matrix4d
#define MatrixXT MatrixXd

#endif
