#pragma once
#include "ofMain.h"

class ambiEncode2DThirdOrder
{
public:
	ambiEncode2DThirdOrder() : output(), azimuth(0), distance(0), gainW(0), gainHO(0) {}

	inline std::array<double, 7> play(const double& pAudioIn, const double& pAzimuth, const double& pDistance)
	{
		azimuth = PI * pAzimuth / 180.0;
		azimuth = (pDistance < 0 ? azimuth + PI : azimuth);
		distance = std::abs(pDistance) + 0.0001;
		gainW = std::atan(distance * PI * 0.5) / (distance * PI * 0.5);
		gainHO = (1 - std::exp(-distance));

		output[0] = gainW * pAudioIn;
		output[1] = std::cos(1 * azimuth) * gainHO * gainW * pAudioIn;
		output[2] = std::sin(1 * azimuth) * gainHO * gainW * pAudioIn;
		output[3] = std::cos(2 * azimuth) * gainHO * gainW * pAudioIn;
		output[4] = std::sin(2 * azimuth) * gainHO * gainW * pAudioIn;
		output[5] = std::cos(3 * azimuth) * gainHO * gainW * pAudioIn;
		output[6] = std::sin(3 * azimuth) * gainHO * gainW * pAudioIn;
		return output;
	}

private:
	double azimuth, distance, gainW, gainHO;
	std::array<double, 7> output;
};

class ambiDecode2DThirdOrder
{
public:
	ambiDecode2DThirdOrder() : output(0), azimuth(), amplitude() {}

	inline double play(const std::array<double, 7>& pDecodeIn, const double& pAzimuth)
	{
		size_t order = 3;
		output = 0.5 * pDecodeIn[0];
		output += std::cos(1 * pAzimuth) * weight2D[order - 1][1] * pDecodeIn[1];
		output += std::sin(1 * pAzimuth) * weight2D[order - 1][2] * pDecodeIn[2];
		output += std::cos(2 * pAzimuth) * weight2D[order - 1][3] * pDecodeIn[3];
		output += std::sin(2 * pAzimuth) * weight2D[order - 1][4] * pDecodeIn[4];
		output += std::cos(3 * pAzimuth) * weight2D[order - 1][5] * pDecodeIn[5];
		output += std::sin(3 * pAzimuth) * weight2D[order - 1][6] * pDecodeIn[6];
		return output * norm2D[order - 1];
	}

private:
	double output, azimuth, amplitude;
    
    std::array<double, 12>
    norm2D =
        { 1.0,      0.75,     0.625,    0.546875,  0.492188,   0.451172,   0.418945,    0.392761,    0.370941,     0.352394,      0.336376,      0.322360       };

    std::array<std::array<double, 12>, 13>
    weight2D =
    { {
        { 0.5,      0.0,      0.0,      0.0,       0.0,        0.0,        0.0,         0.0,         0.0,          0.0,           0.0,           0.0            },
        { 0.666667, 0.166667, 0.0,      0.0,       0.0,        0.0 ,       0.0,         0.0,         0.0,          0.0,           0.0,           0.0            },
        { 0.75,     0.3,      0.05,     0.0,       0.0,        0.0,        0.0,         0.0,         0.0,          0.0,           0.0,           0.0            },
        { 0.75,     0.3,      0.05,     0.0,       0.0,        0.0,        0.0,         0.0,         0.0,          0.0,           0.0,           0.0            },
        { 0.8,      0.4,      0.114286, 0.0142857, 0.0,        0.0,        0.0,         0.0,         0.0,          0.0,           0.0,           0.0            },
        { 0.833333, 0.47619,  0.178571, 0.0396825, 0.00396825, 0.0,        0.0,         0.0,         0.0,          0.0,           0.0,           0.0            },
        { 0.857143, 0.535714, 0.238095, 0.0714286, 0.012987,   0.00108225, 0.0,         0.0,         0.0,          0.0,           0.0,           0.0            },
        { 0.875,    0.583333, 0.291667, 0.1060601, 0.0265152,  0.00407925, 0.000291375, 0.0,         0.0,          0.0,           0.0,           0.0            },
        { 0.888889, 0.622222, 0.339394, 0.141414,  0.043512,   0.009324,   0.0012432,   0.0000777,   0.0,          0.0,           0.0,           0.0            },
        { 0.9,      0.654545, 0.381818, 0.176224,  0.0629371,  0.0167832,  0.00314685,  0.000370218, 0.0000205677, 0.0,           0.0,           0.0            },
        { 0.909091, 0.681818, 0.41958,  0.20979,   0.0839161,  0.0262238,  0.0061703,   0.00102838,  0.000108251,  0.00000541254, 0.0,           0.0            },
        { 0.916667, 0.705128, 0.453297, 0.241758,  0.105769,   0.0373303,  0.0103695,   0.00218306,  0.000327459,  0.0000311866,  0.00000141757, 0.0            },
        { 0.923077, 0.725275, 0.483516, 0.271978,  0.12799,    0.0497738,  0.015718,    0.00392951,  0.000748478,  0.000102065,   0.00000887523, 0.000000369801 },
    } };

};
