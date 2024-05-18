#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>

// for testing: remove #include <dergeraet/config.hpp> from CalculateMoments_<<theory>>.cpp and add the following statement
// #include "test_def.hpp"

// terminal commands:
/*
g++ -c -std=c++17 -O2 CalculateMoments_<<theory>>.cpp
g++ -c -std=c++17 -O2 TestCalculateMoments.cpp
g++ TestCalculateMoments.o CalculateMoments_<<theory>>.o -o TestMoments
./TestMoments

e.g. for 3333333:
g++ -c -std=c++17 -O2 CalculateMoments_3333333.cpp
g++ -c -std=c++17 -O2 TestCalculateMoments.cpp
g++ TestCalculateMoments.o CalculateMoments_3333333.o -o TestMoments
./TestMoments
*/

#include "../CalculateMoments_generated/CalculateMoments_Full2.cpp"
#include "../CalculateMoments_generated/CalculateMoments_3333333.cpp"
#include "../CalculateMoments_generated/CalculateMoments_444444444.cpp"
#include "../CalculateMoments_generated/CalculateMoments_Full10.cpp"

double runde (double value, int precision)
{
    double multiplier = std::pow(10, precision);
    return (double)((int)(value * multiplier + 0.5f))/multiplier;
}

void max_norm (const std::vector<double> moments, const std::vector<double> orig) {
    double max = 0;
    double pos;
    for(int i = 0; i < orig.size(); i++) {
        double zsp = abs(orig[i]-moments[i]);
        if(zsp > max) {
            max = zsp;
            pos = i;
        }
    }
    std::cout << "max_norm (discontinuous f only!) = " << max << std::endl;
    std::cout << "pos = " << pos << std::endl;
}

void calculate_moments (const std::vector<double> orig, int theory) {
    std::vector<double> moments;
    moments.resize(orig.size() + 5);

    std::vector<double> coeffs; // aktuell egal
    config_t<double> conf;
    conf.Nx = conf.Ny = conf.Nz = 1;  // Number of grid points in physical space.
    conf.Nu = conf.Nv = conf.Nw = 80;  // Number of quadrature points in velocity space.
    //conf.Nt = 50;  // Number of time-steps.
    //conf.dt = 1;  // Time-step size.

    // Dimensions of physical domain.
    conf.x_min = conf.y_min = conf.z_min = 0; // aktuell egal
    conf.x_max = conf.y_max = conf.z_max = 1; // aktuell egal

    // Integration limits for velocity space.
    conf.u_min = conf.v_min = conf.w_min = -20; // ausprobieren
    conf.u_max = conf.v_max = conf.w_max = 20;

    // Grid-sizes and their reciprocals.
    conf.dx = conf.dy = conf.dz = 1; // aktuell egal
    //conf.dx_inv = conf.dy_inv = conf.dz_inv = 1;
    //conf.Lx = conf.Ly = conf.Lz = 1;
    //conf.Lx_inv = conf.Ly_inv = conf.Lz_inv = 1;


    conf.du = (conf.u_max-conf.u_min) / conf.Nu; 
    conf.dv = (conf.v_max-conf.v_min) / conf.Nv;
    conf.dw = (conf.w_max-conf.w_min) / conf.Nw;

    size_t n = 1; // aktuell egal
    size_t l = 1; // aktuell egal

    const size_t order = 1; // aktuell egal

    std::ofstream outputFile1("../Moments/Full2.txt");
    std::ofstream outputFile2("../Moments/3333333.txt");
    std::ofstream outputFile3("../Moments/444444444.txt");
    std::ofstream outputFile4("../Moments/Full10.txt");

    switch (theory)
    {
    case 1:
        cm_21100::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        break;
    case 2:
        cm_3333333::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        break;
    case 3:
        cm_444444444::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        break;
    case 4:
        cm_1099887766554433221100::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        break;

    case 5: //Output
        moments.resize(40);
        cm_21100::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        if (!outputFile1) {
            std::cout << "Could not open output file." << std::endl;
        }

        for(int i = 0; i < moments.size(); i++) {
            if(abs(moments[i]) < 1) {
                outputFile1 << "moments[" << i << "] = "  << runde(moments[i],6) << ";" << std::endl;
            } else {
                outputFile1 << "moments[" << i << "] = "  << moments[i] << ";" << std::endl;
            }
        }
        outputFile1.close();

        for(int i = 0; i < moments.size(); i++) {
            moments[i] = 0;
        }

        moments.resize(201);
        cm_3333333::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        if (!outputFile2) {
            std::cout << "Could not open output file." << std::endl;
        }

        for(int i = 0; i < moments.size(); i++) {
            if(abs(moments[i]) < 1) {
                outputFile2 << "moments[" << i << "] = "  << runde(moments[i],6) << ";" << std::endl;
            } else {
                outputFile2 << "moments[" << i << "] = "  << moments[i] << ";" << std::endl;
            }
        }
        outputFile2.close();

        for(int i = 0; i < moments.size(); i++) {
            moments[i] = 0;
        }

        moments.resize(410);
        cm_444444444::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        if (!outputFile3) {
            std::cout << "Could not open output file." << std::endl;
        }

        for(int i = 0; i < moments.size(); i++) {
            if(abs(moments[i]) < 1) {
                outputFile3 << "moments[" << i << "] = "  << runde(moments[i],6) << ";" << std::endl;
            } else {
                outputFile3 << "moments[" << i << "] = "  << moments[i] << ";" << std::endl;
            }
        }
        outputFile3.close();

        for(int i = 0; i < moments.size(); i++) {
            moments[i] = 0;
        }
        moments.resize(1775);
        cm_1099887766554433221100::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        if (!outputFile4) {
            std::cout << "Could not open output file." << std::endl;
        }

        for(int i = 0; i < moments.size(); i++) {
            if(abs(moments[i]) < 1) {
                outputFile4 << "moments[" << i << "] = " << runde(moments[i],6) << ";" << std::endl;
            } else {
                outputFile4 << "moments[" << i << "] = "  << moments[i] << ";" << std::endl;
            }
        }
        outputFile4.close();
        break;
    default:
        std::cout << "No valid number, retry." << std::endl;
        break;
    }

    for(int i = 0; i < moments.size(); i++) {
        std::cout << "moments[" << i << "] = ";
        if(abs(moments[i]) < 1) {
            std::cout << runde(moments[i],6) << ";" << std::endl;
        } else {
            std::cout << moments[i] << ";" << std::endl;
        }
    }

    if(moments[2] + 0.191477 < 0.001) {
        max_norm(moments, orig);
    }
}

void Full2 () {
    std::vector<double> orig;
    orig.resize(35);
    orig[0] = 1.0;
    orig[1] = 0;
    orig[2] = -0.1914778361;
    orig[3] = 0;
    orig[4] = 0;
    orig[5] = 0;
    orig[6] = -0.2797288653;
    orig[7] = -0.02153353809;
    orig[8] = -0.03936924771;
    orig[9] = -0.02205882353;
    orig[10] = 0;
    orig[11] = -0.6749903882;
    orig[12] = -0.2292420186;
    orig[13] = -0.8823529412;
    orig[14] = 0.008669806723;
    orig[15] = 0.002702978232;
    orig[16] = -0.3849241224;
    orig[17] = -0.3207534169;
    orig[18] = -0.281075134;
    orig[19] = -0.232727199;
    orig[20] = -0.03404751323;
    orig[21] = -0.1554049252;
    orig[22] = -0.02637309034;
    orig[23] = -0.03522511637;
    orig[24] = 0.03404751323; 
    orig[25] = -0.232727199;
    orig[26] = 0.03371206157;
    orig[27] = -0.006193301176;
    orig[28] = 0.01667073307;
    orig[29] = 0.002340847815;
    orig[30] = 0.3411191802;
    orig[31] = 0.02809017378;
    orig[32] = 0.5398344391;
    orig[33] = -0.03715980706;
    orig[34] = 0.6971373398;

    calculate_moments(orig, 1);
}

void gauss3333333 () {
    std::vector<double> orig;
    orig.resize(196);
    orig[0] = 1.;
    orig[1] = 0.;
    orig[2] = -0.1914778361;
    orig[3] = -0.07776670619;
    orig[4] = 0.;
    orig[5] = 0.;
    orig[6] = 0.;
    orig[7] = -0.2797288653;
    orig[8] = -0.02153353809;
    orig[9] = -0.03936924771;
    orig[10] = -0.2589288998;
    orig[11] = -0.04296866976;
    orig[12] = -0.02987552558;
    orig[13] = -0.09037523194;
    orig[14] = -0.05899691516;
    orig[15] = 0.0101739669;
    orig[16] = -0.02205882353;
    orig[17] = 0.;
    orig[18] = -0.6749903882;
    orig[19] = -0.2292420186;
    orig[20] = -0.8823529412;
    orig[21] = 0.008669806723;
    orig[22] = 0.002702978232;
    orig[23] = -0.3849241224;
    orig[24] = -0.3207534169;
    orig[25] = -0.281075134;
    orig[26] = 0.02615285447;
    orig[27] = 0.006399698462;
    orig[28] = -0.04958220487;
    orig[29] = -0.3566937525;
    orig[30] = 0.3287590709;
    orig[31] = 0.02533448439;
    orig[32] = 0.01021702081;
    orig[33] = 0.1453688896;
    orig[34] = -0.3544003752;
    orig[35] = 0.6515752971;
    orig[36] = -0.232727199;
    orig[37] = -0.03404751323;
    orig[38] = -0.1554049252;
    orig[39] = -0.02637309034;
    orig[40] = -0.03522511637;
    orig[41] = 0.03404751323;
    orig[42] = -0.232727199;
    orig[43] = -0.1045208586;
    orig[44] = -0.05664759408;
    orig[45] = -0.03720027817;
    orig[46] = -0.0400396219;
    orig[47] = -0.01767902012;
    orig[48] = 0.05062878721;
    orig[49] = -0.2292226575;
    orig[50] = 0.05120700735;
    orig[51] = -0.0734066454;
    orig[52] = 0.09722809503;
    orig[53] = -0.04652701591;
    orig[54] = 0.01317974698;
    orig[55] = 0.05720743509;
    orig[56] = -0.1574815177;
    orig[57] = 0.1303149305;
    orig[58] = -0.0840941855;
    orig[59] = 0.1661233175;
    orig[60] = -0.04666300145;
    orig[61] = 0.0416457073;
    orig[62] = 0.05513042377;
    orig[63] = -0.07740683057;
    orig[64] = 0.03371206157;
    orig[65] = -0.006193301176;
    orig[66] = 0.01667073307;
    orig[67] = 0.002340847815;
    orig[68] = 0.3411191802;
    orig[69] = 0.02809017378;
    orig[70] = 0.5398344391;
    orig[71] = -0.03715980706;
    orig[72] = 0.6971373398;
    orig[73] = 0.01068074081;
    orig[74] = -0.01071867307;
    orig[75] = 0.01132513973;
    orig[76] = 0.00421274158;
    orig[77] = 0.3884959124;
    orig[78] = 0.05242441308;
    orig[79] = 0.6423436284;
    orig[80] = -0.06969079282;
    orig[81] = 0.8313713893;
    orig[82] = -0.0291274452;
    orig[83] = -0.01403138406;
    orig[84] = -0.0005452442944;
    orig[85] = 0.005784771678;
    orig[86] = 0.298182948;
    orig[87] = 0.07499720484;
    orig[88] = 0.5405002852;
    orig[89] = -0.1002251336;
    orig[90] = 0.7023710683;
    orig[91] = -0.06395502996;
    orig[92] = -0.01577908911;
    orig[93] = -0.01094058235;
    orig[94] = 0.006906920487;
    orig[95] = 0.1473993937;
    orig[96] = 0.09381310375;
    orig[97] = 0.3456185213;
    orig[98] = -0.1260876545;
    orig[99] = 0.4529599443;
    orig[100] = 0.1492305857;
    orig[101] = -0.008238738477;
    orig[102] = 0.1148884017;
    orig[103] = 0.002002794896;
    orig[104] = 0.1126485232;
    orig[105] = 0.00403121335;
    orig[106] = 0.02709683196;
    orig[107] = -0.006258734049;
    orig[108] = 0.07106519355;
    orig[109] = 0.007371502848;
    orig[110] = 0.1568834362;
    orig[111] = 0.1617578891;
    orig[112] = -0.01716671366;
    orig[113] = 0.1246645235;
    orig[114] = 0.004129248301;
    orig[115] = 0.1269833543;
    orig[116] = 0.008320917462;
    orig[117] = 0.04128440752;
    orig[118] = -0.0128440633;
    orig[119] = 0.1008141689;
    orig[120] = 0.01493505339;
    orig[121] = 0.2221093861;
    orig[122] = 0.09547126359;
    orig[123] = -0.02713184952;
    orig[124] = 0.07395688551;
    orig[125] = 0.006453120772;
    orig[126] = 0.08523589638;
    orig[127] = 0.0130199412;
    orig[128] = 0.04693099611;
    orig[129] = -0.01997174124;
    orig[130] = 0.1030245873;
    orig[131] = 0.02289729099;
    orig[132] = 0.2259890077;
    orig[133] = -0.009271134161;
    orig[134] = -0.03722074728;
    orig[135] = -0.006161176006;
    orig[136] = 0.008746829592;
    orig[137] = 0.01614489347;
    orig[138] = 0.01767141818;
    orig[139] = 0.04550686777;
    orig[140] = -0.02692301887;
    orig[141] = 0.08442673134;
    orig[142] = 0.03038736431;
    orig[143] = 0.1833251132;
    orig[144] = -0.04212238836;
    orig[145] = -0.001832908558;
    orig[146] = -0.02090703144;
    orig[147] = 0.000761022601;
    orig[148] = -0.008746721917;
    orig[149] = -0.0001754786811;
    orig[150] = -0.1329482441;
    orig[151] = -0.000157930813;
    orig[152] = -0.1975516541;
    orig[153] = 0.001272334661;
    orig[154] = -0.2040361653;
    orig[155] = -0.003248321277;
    orig[156] = -0.2642330028;
    orig[157] = -0.059011728;
    orig[158] = -0.004060074296;
    orig[159] = -0.02916705469;
    orig[160] = 0.001720857319;
    orig[161] = -0.01151232311;
    orig[162] = -0.0004029705366;
    orig[163] = -0.2184531541;
    orig[164] = -0.0003118039298;
    orig[165] = -0.328771149;
    orig[166] = 0.002843480385;
    orig[167] = -0.3358368944;
    orig[168] = -0.007338825634;
    orig[169] = -0.4357772706;
    orig[170] = -0.0558543857;
    orig[171] = -0.00675209299;
    orig[172] = -0.02734877814;
    orig[173] = 0.002926085888;
    orig[174] = -0.009457722945;
    orig[175] = -0.0006962511756;
    orig[176] = -0.2619804117;
    orig[177] = -0.0004489997695;
    orig[178] = -0.4019297823;
    orig[179] = 0.004774802388;
    orig[180] = -0.4038393146;
    orig[181] = -0.01246718383;
    orig[182] = -0.5256708224;
    orig[183] = -0.03655185948;
    orig[184] = -0.009664916752;
    orig[185] = -0.0173993457;
    orig[186] = 0.004289916902;
    orig[187] = -0.003605089226;
    orig[188] = -0.001037862386;
    orig[189] = -0.2601308551;
    orig[190] = -0.000532747683;
    orig[191] = -0.4113951455;
    orig[192] = 0.006907296164;
    orig[193] = -0.4027698992;
    orig[194] = -0.01826028117;
    orig[195] = -0.5270443588;

    calculate_moments(orig, 2);
}

void gauss444444444 () {
    std::vector<double> orig;
    orig.resize(405);
    orig[0] = 1.0;
    orig[1] = 0.0;
    orig[2] = -0.1914778361;
    orig[3] = -0.07776670619;
    orig[4] = 0.04928594337;
    orig[5] = 0.0;
    orig[6] = 0.0;
    orig[7] = 0.0;
    orig[8] = -0.2797288653;
    orig[9] = -0.02153353809;
    orig[10] = -0.03936924771;
    orig[11] = -0.2589288998;
    orig[12] = -0.04296866976;
    orig[13] = -0.02987552558;
    orig[14] = -0.09037523194;
    orig[15] = -0.05899691516;
    orig[16] = 0.0101739669;
    orig[17] = 0.07240877898;
    orig[18] = -0.06820429155;
    orig[19] = 0.05628887971;
    orig[20] = -0.02205882353;
    orig[21] = 0.0;
    orig[22] = -0.6749903882;
    orig[23] = -0.2292420186;
    orig[24] = -0.8823529412;
    orig[25] = 0.008669806723;
    orig[26] = 0.002702978232;
    orig[27] = -0.3849241224;
    orig[28] = -0.3207534169;
    orig[29] = -0.281075134;
    orig[30] = 0.02615285447;
    orig[31] = 0.006399698462;
    orig[32] = -0.04958220487;
    orig[33] = -0.3566937525;
    orig[34] = 0.3287590709;
    orig[35] = 0.02533448439;
    orig[36] = 0.01021702081;
    orig[37] = 0.1453688896;
    orig[38] = -0.3544003752;
    orig[39] = 0.6515752971;
    orig[40] = 0.01460642842;
    orig[41] = 0.01358995858;
    orig[42] = 0.1923122618;
    orig[43] = -0.3274888891;
    orig[44] = 0.6920171203;
    orig[45] = -0.232727199;
    orig[46] = -0.03404751323;
    orig[47] = -0.1554049252;
    orig[48] = -0.02637309034;
    orig[49] = -0.03522511637;
    orig[50] = 0.03404751323;
    orig[51] = -0.232727199;
    orig[52] = -0.1045208586;
    orig[53] = -0.05664759408;
    orig[54] = -0.03720027817;
    orig[55] = -0.0400396219;
    orig[56] = -0.01767902012;
    orig[57] = 0.05062878721;
    orig[58] = -0.2292226575;
    orig[59] = 0.05120700735;
    orig[60] = -0.0734066454;
    orig[61] = 0.09722809503;
    orig[62] = -0.04652701591;
    orig[63] = 0.01317974698;
    orig[64] = 0.05720743509;
    orig[65] = -0.1574815177;
    orig[66] = 0.1303149305;
    orig[67] = -0.0840941855;
    orig[68] = 0.1661233175;
    orig[69] = -0.04666300145;
    orig[70] = 0.0416457073;
    orig[71] = 0.05513042377;
    orig[72] = -0.07740683057;
    orig[73] = 0.1170314801;
    orig[74] = -0.08922185323;
    orig[75] = 0.1578386076;
    orig[76] = -0.04185590593;
    orig[77] = 0.06069039541;
    orig[78] = 0.04649586126;
    orig[79] = -0.01762966016;
    orig[80] = 0.03371206157;
    orig[81] = -0.006193301176;
    orig[82] = 0.01667073307;
    orig[83] = 0.002340847815;
    orig[84] = 0.3411191802;
    orig[85] = 0.02809017378;
    orig[86] = 0.5398344391;
    orig[87] = -0.03715980706;
    orig[88] = 0.6971373398;
    orig[89] = 0.01068074081;
    orig[90] = -0.01071867307;
    orig[91] = 0.01132513973;
    orig[92] = 0.00421274158;
    orig[93] = 0.3884959124;
    orig[94] = 0.05242441308;
    orig[95] = 0.6423436284;
    orig[96] = -0.06969079282;
    orig[97] = 0.8313713893;
    orig[98] = -0.0291274452;
    orig[99] = -0.01403138406;
    orig[100] = -0.0005452442944;
    orig[101] = 0.005784771678;
    orig[102] = 0.298182948;
    orig[103] = 0.07499720484;
    orig[104] = 0.5405002852;
    orig[105] = -0.1002251336;
    orig[106] = 0.7023710683;
    orig[107] = -0.06395502996;
    orig[108] = -0.01577908911;
    orig[109] = -0.01094058235;
    orig[110] = 0.006906920487;
    orig[111] = 0.1473993937;
    orig[112] = 0.09381310375;
    orig[113] = 0.3456185213;
    orig[114] = -0.1260876545;
    orig[115] = 0.4529599443;
    orig[116] = -0.0822145994;
    orig[117] = -0.01591225811;
    orig[118] = -0.01561639634;
    orig[119] = 0.007526069434;
    orig[120] = -0.003109386103;
    orig[121] = 0.1078359833;
    orig[122] = 0.1457804176;
    orig[123] = -0.1458352959;
    orig[124] = 0.1962410854;
    orig[125] = 0.1492305857;
    orig[126] = -0.008238738477;
    orig[127] = 0.1148884017;
    orig[128] = 0.002002794896;
    orig[129] = 0.1126485232;
    orig[130] = 0.00403121335;
    orig[131] = 0.02709683196;
    orig[132] = -0.006258734049;
    orig[133] = 0.07106519355;
    orig[134] = 0.007371502848;
    orig[135] = 0.1568834362;
    orig[136] = 0.1617578891;
    orig[137] = -0.01716671366;
    orig[138] = 0.1246645235;
    orig[139] = 0.004129248301;
    orig[140] = 0.1269833543;
    orig[141] = 0.008320917462;
    orig[142] = 0.04128440752;
    orig[143] = -0.0128440633;
    orig[144] = 0.1008141689;
    orig[145] = 0.01493505339;
    orig[146] = 0.2221093861;
    orig[147] = 0.09547126359;
    orig[148] = -0.02713184952;
    orig[149] = 0.07395688551;
    orig[150] = 0.006453120772;
    orig[151] = 0.08523589638;
    orig[152] = 0.0130199412;
    orig[153] = 0.04693099611;
    orig[154] = -0.01997174124;
    orig[155] = 0.1030245873;
    orig[156] = 0.02289729099;
    orig[157] = 0.2259890077;
    orig[158] = -0.009271134161;
    orig[159] = -0.03722074728;
    orig[160] = -0.006161176006;
    orig[161] = 0.008746829592;
    orig[162] = 0.01614489347;
    orig[163] = 0.01767141818;
    orig[164] = 0.04550686777;
    orig[165] = -0.02692301887;
    orig[166] = 0.08442673134;
    orig[167] = 0.03038736431;
    orig[168] = 0.1833251132;
    orig[169] = -0.1128314519;
    orig[170] = -0.04667939529;
    orig[171] = -0.08523211625;
    orig[172] = 0.01082930416;
    orig[173] = -0.05209076468;
    orig[174] = 0.02191040418;
    orig[175] = 0.03963506867;
    orig[176] = -0.03313508495;
    orig[177] = 0.05486502874;
    orig[178] = 0.03675216485;
    orig[179] = 0.1158698955;
    orig[180] = -0.04212238836;
    orig[181] = -0.001832908558;
    orig[182] = -0.02090703144;
    orig[183] = 0.000761022601;
    orig[184] = -0.008746721917;
    orig[185] = -0.0001754786811;
    orig[186] = -0.1329482441;
    orig[187] = -0.000157930813;
    orig[188] = -0.1975516541;
    orig[189] = 0.001272334661;
    orig[190] = -0.2040361653;
    orig[191] = -0.003248321277;
    orig[192] = -0.2642330028;
    orig[193] = -0.059011728;
    orig[194] = -0.004060074296;
    orig[195] = -0.02916705469;
    orig[196] = 0.001720857319;
    orig[197] = -0.01151232311;
    orig[198] = -0.0004029705366;
    orig[199] = -0.2184531541;
    orig[200] = -0.0003118039298;
    orig[201] = -0.328771149;
    orig[202] = 0.002843480385;
    orig[203] = -0.3358368944;
    orig[204] = -0.007338825634;
    orig[205] = -0.4357772706;
    orig[206] = -0.0558543857;
    orig[207] = -0.00675209299;
    orig[208] = -0.02734877814;
    orig[209] = 0.002926085888;
    orig[210] = -0.009457722945;
    orig[211] = -0.0006962511756;
    orig[212] = -0.2619804117;
    orig[213] = -0.0004489997695;
    orig[214] = -0.4019297823;
    orig[215] = 0.004774802388;
    orig[216] = -0.4038393146;
    orig[217] = -0.01246718383;
    orig[218] = -0.5256708224;
    orig[219] = -0.03655185948;
    orig[220] = -0.009664916752;
    orig[221] = -0.0173993457;
    orig[222] = 0.004289916902;
    orig[223] = -0.003605089226;
    orig[224] = -0.001037862386;
    orig[225] = -0.2601308551;
    orig[226] = -0.000532747683;
    orig[227] = -0.4113951455;
    orig[228] = 0.006907296164;
    orig[229] = -0.4027698992;
    orig[230] = -0.01826028117;
    orig[231] = -0.5270443588;
    orig[232] = -0.008954621442;
    orig[233] = -0.01255613704;
    orig[234] = -0.003239877821;
    orig[235] = 0.005719625922;
    orig[236] = 0.004121421555;
    orig[237] = -0.001407815779;
    orig[238] = -0.2213532596;
    orig[239] = -0.0005335803384;
    orig[240] = -0.3684832657;
    orig[241] = 0.009078361794;
    orig[242] = -0.3454341023;
    orig[243] = -0.02432088761;
    orig[244] = -0.4562811891;
    orig[245] = -0.07075655194;
    orig[246] = -0.001003272408;
    orig[247] = -0.05460041117;
    orig[248] = 0.00007192195037;
    orig[249] = -0.04972740114;
    orig[250] = 0.00009835544702;
    orig[251] = -0.04910816705;
    orig[252] = 0.00006009788482;
    orig[253] = -0.00818361885;
    orig[254] = 0.00001336232131;
    orig[255] = -0.02208501971;
    orig[256] = -0.0002969349094;
    orig[257] = -0.04208245959;
    orig[258] = 0.0006889311833;
    orig[259] = -0.07805010194;
    orig[260] = -0.1197779287;
    orig[261] = -0.002445054034;
    orig[262] = -0.09244925279;
    orig[263] = 0.0001677856511;
    orig[264] = -0.0832362628;
    orig[265] = 0.0002438430888;
    orig[266] = -0.08257202541;
    orig[267] = 0.0001352787912;
    orig[268] = -0.01572105252;
    orig[269] = 0.00004549261884;
    orig[270] = -0.04080408747;
    orig[271] = -0.000726150385;
    orig[272] = -0.07882317665;
    orig[273] = 0.001664277535;
    orig[274] = -0.1458209967;
    orig[275] = -0.1448028731;
    orig[276] = -0.004451298321;
    orig[277] = -0.1118577237;
    orig[278] = 0.0002912153945;
    orig[279] = -0.09881191849;
    orig[280] = 0.0004517989715;
    orig[281] = -0.09878903748;
    orig[282] = 0.0002250206551;
    orig[283] = -0.02242114953;
    orig[284] = 0.000107392561;
    orig[285] = -0.05520164155;
    orig[286] = -0.001326725088;
    orig[287] = -0.1086885449;
    orig[288] = 0.003001925246;
    orig[289] = -0.200379949;
    orig[290] = -0.1404833733;
    orig[291] = -0.00695111918;
    orig[292] = -0.1087634511;
    orig[293] = 0.0004315230133;
    orig[294] = -0.09278072147;
    orig[295] = 0.0007183734824;
    orig[296] = -0.09414830474;
    orig[297] = 0.0003167095405;
    orig[298] = -0.02727761449;
    orig[299] = 0.0002077898625;
    orig[300] = -0.06243977762;
    orig[301] = -0.002079546945;
    orig[302] = -0.126346003;
    orig[303] = 0.004642198512;
    orig[304] = -0.231825553;
    orig[305] = -0.1103420877;
    orig[306] = -0.009830623884;
    orig[307] = -0.08593905603;
    orig[308] = 0.0005759274967;
    orig[309] = -0.06790715099;
    orig[310] = 0.001034953423;
    orig[311] = -0.07131092423;
    orig[312] = 0.0003966331868;
    orig[313] = -0.0299280144;
    orig[314] = 0.0003531312686;
    orig[315] = -0.06180677118;
    orig[316] = -0.002952445476;
    orig[317] = -0.1302557731;
    orig[318] = 0.006497832205;
    orig[319] = -0.2373805607;
    orig[320] = 0.02183503059;
    orig[321] = -0.0002355826294;
    orig[322] = 0.01207226884;
    orig[323] = 0.00003686801067;
    orig[324] = 0.007554252229;
    orig[325] = 0.00002744429082;
    orig[326] = 0.00350259698;
    orig[327] = -0.00001630493573;
    orig[328] = 0.0390305248;
    orig[329] = -0.0001965743479;
    orig[330] = 0.05581026511;
    orig[331] = 0.0001402610305;
    orig[332] = 0.05642564086;
    orig[333] = -6.397563587e-6;
    orig[334] = 0.0601725929;
    orig[335] = -0.0001803002558;
    orig[336] = 0.07641376394;
    orig[337] = 0.04111161613;
    orig[338] = -0.0006070494188;
    orig[339] = 0.0226529847;
    orig[340] = 0.00009774994397;
    orig[341] = 0.014331844;
    orig[342] = 0.00006918167432;
    orig[343] = 0.006557072245;
    orig[344] = -0.0000416134585;
    orig[345] = 0.07995267961;
    orig[346] = -0.0005117603096;
    orig[347] = 0.1145145458;
    orig[348] = 0.000365806723;
    orig[349] = 0.115530038;
    orig[350] = -0.0000180929096;
    orig[351] = 0.123410417;
    orig[352] = -0.0004672087957;
    orig[353] = 0.156548361;
    orig[354] = 0.05563723022;
    orig[355] = -0.001161759382;
    orig[356] = 0.03050083249;
    orig[357] = 0.0001926341473;
    orig[358] = 0.01962524474;
    orig[359] = 0.000129288445;
    orig[360] = 0.008796934109;
    orig[361] = -0.00007882728101;
    orig[362] = 0.1203204516;
    orig[363] = -0.0009899760802;
    orig[364] = 0.172712391;
    orig[365] = 0.0007089420353;
    orig[366] = 0.173754869;
    orig[367] = -0.00003787940109;
    orig[368] = 0.1860358169;
    orig[369] = -0.000899419217;
    orig[370] = 0.2356646134;
    orig[371] = 0.0613949657;
    orig[372] = -0.001897986408;
    orig[373] = 0.03338205079;
    orig[374] = 0.0003243315191;
    orig[375] = 0.02207824469;
    orig[376] = 0.0002058407648;
    orig[377] = 0.009570495304;
    orig[378] = -0.000127377292;
    orig[379] = 0.1526147673;
    orig[380] = -0.001635641387;
    orig[381] = 0.2197228703;
    orig[382] = 0.001173551592;
    orig[383] = 0.2202185834;
    orig[384] = -0.00006751237343;
    orig[385] = 0.2365378024;
    orig[386] = -0.001478537444;
    orig[387] = 0.2991088857;
    orig[388] = 0.05696722462;
    orig[389] = -0.002796817596;
    orig[390] = 0.03052239837;
    orig[391] = 0.0004929587165;
    orig[392] = 0.02120468919;
    orig[393] = 0.000294915163;
    orig[394] = 0.008653532925;
    orig[395] = -0.0001855055289;
    orig[396] = 0.1719710891;
    orig[397] = -0.002438826327;
    orig[398] = 0.2486112456;
    orig[399] = 0.001753280092;
    orig[400] = 0.2478925373;
    orig[401] = -0.0001082777655;
    orig[402] = 0.267461933;
    orig[403] = -0.002193011377;
    orig[404] = 0.3374265533;

    calculate_moments(orig, 3);
}

void Full10 () {
    std::vector<double> orig;
    orig.resize(1771);
    calculate_moments(orig, 4);
}


int main () {
    bool cont = true;
    int choice; 
    char con;
    while (cont) {

        std::cout << "Select a theory (select number): 1) Full 2; 2) 3333333; 3) 444444444; 4) Full10; 5) Everything (safe in Moments)" << std::endl;
        std::cin >> choice;

        std::vector<double> orig;
        switch (choice)
        {
            case 1:
                Full2();
                break;
            case 2:
                gauss3333333();
                break;
            case 3:
                gauss444444444();
                break;
            case 4:
                Full10();
                break;
            case 5:
                calculate_moments(orig, 5);
                break;
            default:
                break;
        }
        std::cout << "Continue? [y/n]" << std::endl;
        std::cin >> con;
        if (con == 'n') cont = false;
    }
    return 0;
}