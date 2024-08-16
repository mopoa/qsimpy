// Generated from qBraid v0.7.1
OPENQASM 3.0;
include 'stdgates.inc';
// Generated from qiskit v0.24.0 file qiskit/qasm/libs/qelib1.inc

// QASM 3 equivalent definitions of qelib1.inc gates

// idle gate (identity) with length gamma*sqglen
 
// generic single qubit gate

// inverse sqrt(X)

// controlled-sqrt(X)

// controlled phase rotation 

// controlled-U

// two-qubit XX rotation
// gate rxx(theta) a, b
// {
//   u3(pi/2, theta, 0) a;
//   h b;
//   cx a,b;
//   u1(-theta) b;
//   cx a,b;
//   h b;
//   ******** why type error ********
//   // u2(-pi, pi-theta) b;
// }

// two-qubit ZZ rotation

// relative-phase CCX

// relative-phase 3-controlled X gate

// 3-controlled X gate

// 3-controlled sqrt(X) gate, this equals the C3X gate 
// where the CU1 rotations are -pi/8 not -pi/4

// 4-controlled X gate 

qubit[48] q;

qubit[1] psi;
bit[48] c;
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
h q[21];
h q[22];
h q[23];
h q[24];
h q[25];
h q[26];
h q[27];
h q[28];
h q[29];
h q[30];
h q[31];
h q[32];
h q[33];
h q[34];
h q[35];
h q[36];
h q[37];
h q[38];
h q[39];
h q[40];
h q[41];
h q[42];
h q[43];
h q[44];
h q[45];
h q[46];
h q[47];
x psi[0];
cp(0.20471795265251574) psi[0],q[0];
cp(0.4094359053050315) psi[0],q[1];
cp(0.818871810610063) psi[0],q[2];
cp(1.637743621220126) psi[0],q[3];
cp(-3.0076980647393343) psi[0],q[4];
cp(0.2677891777009175) psi[0],q[5];
cp(0.535578355401835) psi[0],q[6];
cp(1.07115671080367) psi[0],q[7];
cp(2.14231342160734) psi[0],q[8];
cp(-1.9985584639649066) psi[0],q[9];
cp(2.286068379249773) psi[0],q[10];
cp(-1.7110485486800402) psi[0],q[11];
cp(2.861088209819506) psi[0],q[12];
cp(-0.5610088875405743) psi[0],q[13];
cp(-1.1220177750811486) psi[0],q[14];
cp(-2.2440355501622973) psi[0],q[15];
cp(1.795114206854992) psi[0],q[16];
cp(-2.6929568934696024) psi[0],q[17];
cp(0.8972715202403814) psi[0],q[18];
cp(1.7945430404807627) psi[0],q[19];
cp(-2.6940992262180607) psi[0],q[20];
cp(0.8949868547434644) psi[0],q[21];
cp(1.7899737094869288) psi[0],q[22];
cp(-2.7032378882057286) psi[0],q[23];
cp(0.8767095307681287) psi[0],q[24];
cp(1.7534190615362575) psi[0],q[25];
cp(-2.7763471841070713) psi[0],q[26];
cp(0.7304909389654439) psi[0],q[27];
cp(1.4609818779308879) psi[0],q[28];
cp(2.9219637558617757) psi[0],q[29];
cp(-0.43925779545603444) psi[0],q[30];
cp(-0.8785155909120689) psi[0],q[31];
cp(-1.7570311818241378) psi[0],q[32];
cp(2.7691229435313107) psi[0],q[33];
cp(-0.7449394201169645) psi[0],q[34];
cp(-1.489878840233929) psi[0],q[35];
cp(-2.979757680467858) psi[0],q[36];
cp(0.3236699462438703) psi[0],q[37];
cp(0.6473398924877406) psi[0],q[38];
cp(1.2946797849754812) psi[0],q[39];
cp(2.5893595699509624) psi[0],q[40];
cp(-1.1044661672776617) psi[0],q[41];
cp(-2.2089323345553233) psi[0],q[42];
cp(1.8653206380689396) psi[0],q[43];
cp(-13*pi/16) psi[0],q[44];
cp(3*pi/8) psi[0],q[45];
cp(3*pi/4) psi[0],q[46];
cp(-pi/2) psi[0],q[47];
swap q[0],q[47];
h q[0];
swap q[1],q[46];
cp(-pi/2) q[1],q[0];
h q[1];
swap q[10],q[37];
swap q[11],q[36];
swap q[12],q[35];
swap q[13],q[34];
swap q[14],q[33];
swap q[15],q[32];
swap q[16],q[31];
swap q[17],q[30];
swap q[18],q[29];
swap q[19],q[28];
swap q[2],q[45];
cp(-pi/4) q[2],q[0];
cp(-pi/2) q[2],q[1];
h q[2];
swap q[20],q[27];
swap q[21],q[26];
swap q[22],q[25];
swap q[23],q[24];
swap q[3],q[44];
cp(-pi/8) q[3],q[0];
cp(-pi/4) q[3],q[1];
cp(-pi/2) q[3],q[2];
h q[3];
swap q[4],q[43];
cp(-pi/16) q[4],q[0];
cp(-pi/8) q[4],q[1];
cp(-pi/4) q[4],q[2];
cp(-pi/2) q[4],q[3];
h q[4];
swap q[5],q[42];
cp(-pi/32) q[5],q[0];
cp(-pi/16) q[5],q[1];
cp(-pi/8) q[5],q[2];
cp(-pi/4) q[5],q[3];
cp(-pi/2) q[5],q[4];
h q[5];
swap q[6],q[41];
cp(-pi/64) q[6],q[0];
cp(-pi/32) q[6],q[1];
cp(-pi/16) q[6],q[2];
cp(-pi/8) q[6],q[3];
cp(-pi/4) q[6],q[4];
cp(-pi/2) q[6],q[5];
h q[6];
swap q[7],q[40];
cp(-pi/128) q[7],q[0];
cp(-pi/64) q[7],q[1];
cp(-pi/32) q[7],q[2];
cp(-pi/16) q[7],q[3];
cp(-pi/8) q[7],q[4];
cp(-pi/4) q[7],q[5];
cp(-pi/2) q[7],q[6];
h q[7];
swap q[8],q[39];
cp(-pi/256) q[8],q[0];
cp(-pi/128) q[8],q[1];
cp(-pi/64) q[8],q[2];
cp(-pi/32) q[8],q[3];
cp(-pi/16) q[8],q[4];
cp(-pi/8) q[8],q[5];
cp(-pi/4) q[8],q[6];
cp(-pi/2) q[8],q[7];
h q[8];
swap q[9],q[38];
cp(-pi/512) q[9],q[0];
cp(-pi/1024) q[10],q[0];
cp(-pi/2048) q[11],q[0];
cp(-pi/4096) q[12],q[0];
cp(-pi/8192) q[13],q[0];
cp(-pi/16384) q[14],q[0];
cp(-pi/32768) q[15],q[0];
cp(-pi/65536) q[16],q[0];
cp(-pi/131072) q[17],q[0];
cp(-pi/262144) q[18],q[0];
cp(-pi/524288) q[19],q[0];
cp(-pi/1048576) q[20],q[0];
cp(-pi/2097152) q[21],q[0];
cp(-pi/4194304) q[22],q[0];
cp(-pi/8388608) q[23],q[0];
cp(-pi/16777216) q[24],q[0];
cp(-pi/33554432) q[25],q[0];
cp(-pi/67108864) q[26],q[0];
cp(-pi/134217728) q[27],q[0];
cp(-pi/268435456) q[28],q[0];
cp(-pi/536870912) q[29],q[0];
cp(-pi/1073741824) q[30],q[0];
cp(-pi/2147483648) q[31],q[0];
cp(-pi/4294967296) q[32],q[0];
cp(-pi/8589934592) q[33],q[0];
cp(-pi/17179869184) q[34],q[0];
cp(-pi/34359738368) q[35],q[0];
cp(-pi/68719476736) q[36],q[0];
cp(-pi/137438953472) q[37],q[0];
cp(-pi/274877906944) q[38],q[0];
cp(-pi/549755813888) q[39],q[0];
cp(-pi/1099511627776) q[40],q[0];
cp(-pi/2199023255552) q[41],q[0];
cp(0) q[42],q[0];
cp(0) q[43],q[0];
cp(0) q[44],q[0];
cp(0) q[45],q[0];
cp(0) q[46],q[0];
cp(0) q[47],q[0];
cp(-pi/256) q[9],q[1];
cp(-pi/512) q[10],q[1];
cp(-pi/1024) q[11],q[1];
cp(-pi/2048) q[12],q[1];
cp(-pi/4096) q[13],q[1];
cp(-pi/8192) q[14],q[1];
cp(-pi/16384) q[15],q[1];
cp(-pi/32768) q[16],q[1];
cp(-pi/65536) q[17],q[1];
cp(-pi/131072) q[18],q[1];
cp(-pi/262144) q[19],q[1];
cp(-pi/524288) q[20],q[1];
cp(-pi/1048576) q[21],q[1];
cp(-pi/2097152) q[22],q[1];
cp(-pi/4194304) q[23],q[1];
cp(-pi/8388608) q[24],q[1];
cp(-pi/16777216) q[25],q[1];
cp(-pi/33554432) q[26],q[1];
cp(-pi/67108864) q[27],q[1];
cp(-pi/134217728) q[28],q[1];
cp(-pi/268435456) q[29],q[1];
cp(-pi/536870912) q[30],q[1];
cp(-pi/1073741824) q[31],q[1];
cp(-pi/2147483648) q[32],q[1];
cp(-pi/4294967296) q[33],q[1];
cp(-pi/8589934592) q[34],q[1];
cp(-pi/17179869184) q[35],q[1];
cp(-pi/34359738368) q[36],q[1];
cp(-pi/68719476736) q[37],q[1];
cp(-pi/137438953472) q[38],q[1];
cp(-pi/274877906944) q[39],q[1];
cp(-pi/549755813888) q[40],q[1];
cp(-pi/1099511627776) q[41],q[1];
cp(-pi/2199023255552) q[42],q[1];
cp(0) q[43],q[1];
cp(0) q[44],q[1];
cp(0) q[45],q[1];
cp(0) q[46],q[1];
cp(0) q[47],q[1];
cp(-pi/128) q[9],q[2];
cp(-pi/256) q[10],q[2];
cp(-pi/512) q[11],q[2];
cp(-pi/1024) q[12],q[2];
cp(-pi/2048) q[13],q[2];
cp(-pi/4096) q[14],q[2];
cp(-pi/8192) q[15],q[2];
cp(-pi/16384) q[16],q[2];
cp(-pi/32768) q[17],q[2];
cp(-pi/65536) q[18],q[2];
cp(-pi/131072) q[19],q[2];
cp(-pi/262144) q[20],q[2];
cp(-pi/524288) q[21],q[2];
cp(-pi/1048576) q[22],q[2];
cp(-pi/2097152) q[23],q[2];
cp(-pi/4194304) q[24],q[2];
cp(-pi/8388608) q[25],q[2];
cp(-pi/16777216) q[26],q[2];
cp(-pi/33554432) q[27],q[2];
cp(-pi/67108864) q[28],q[2];
cp(-pi/134217728) q[29],q[2];
cp(-pi/268435456) q[30],q[2];
cp(-pi/536870912) q[31],q[2];
cp(-pi/1073741824) q[32],q[2];
cp(-pi/2147483648) q[33],q[2];
cp(-pi/4294967296) q[34],q[2];
cp(-pi/8589934592) q[35],q[2];
cp(-pi/17179869184) q[36],q[2];
cp(-pi/34359738368) q[37],q[2];
cp(-pi/68719476736) q[38],q[2];
cp(-pi/137438953472) q[39],q[2];
cp(-pi/274877906944) q[40],q[2];
cp(-pi/549755813888) q[41],q[2];
cp(-pi/1099511627776) q[42],q[2];
cp(-pi/2199023255552) q[43],q[2];
cp(0) q[44],q[2];
cp(0) q[45],q[2];
cp(0) q[46],q[2];
cp(0) q[47],q[2];
cp(-pi/64) q[9],q[3];
cp(-pi/128) q[10],q[3];
cp(-pi/256) q[11],q[3];
cp(-pi/512) q[12],q[3];
cp(-pi/1024) q[13],q[3];
cp(-pi/2048) q[14],q[3];
cp(-pi/4096) q[15],q[3];
cp(-pi/8192) q[16],q[3];
cp(-pi/16384) q[17],q[3];
cp(-pi/32768) q[18],q[3];
cp(-pi/65536) q[19],q[3];
cp(-pi/131072) q[20],q[3];
cp(-pi/262144) q[21],q[3];
cp(-pi/524288) q[22],q[3];
cp(-pi/1048576) q[23],q[3];
cp(-pi/2097152) q[24],q[3];
cp(-pi/4194304) q[25],q[3];
cp(-pi/8388608) q[26],q[3];
cp(-pi/16777216) q[27],q[3];
cp(-pi/33554432) q[28],q[3];
cp(-pi/67108864) q[29],q[3];
cp(-pi/134217728) q[30],q[3];
cp(-pi/268435456) q[31],q[3];
cp(-pi/536870912) q[32],q[3];
cp(-pi/1073741824) q[33],q[3];
cp(-pi/2147483648) q[34],q[3];
cp(-pi/4294967296) q[35],q[3];
cp(-pi/8589934592) q[36],q[3];
cp(-pi/17179869184) q[37],q[3];
cp(-pi/34359738368) q[38],q[3];
cp(-pi/68719476736) q[39],q[3];
cp(-pi/137438953472) q[40],q[3];
cp(-pi/274877906944) q[41],q[3];
cp(-pi/549755813888) q[42],q[3];
cp(-pi/1099511627776) q[43],q[3];
cp(-pi/2199023255552) q[44],q[3];
cp(0) q[45],q[3];
cp(0) q[46],q[3];
cp(0) q[47],q[3];
cp(-pi/32) q[9],q[4];
cp(-pi/64) q[10],q[4];
cp(-pi/128) q[11],q[4];
cp(-pi/256) q[12],q[4];
cp(-pi/512) q[13],q[4];
cp(-pi/1024) q[14],q[4];
cp(-pi/2048) q[15],q[4];
cp(-pi/4096) q[16],q[4];
cp(-pi/8192) q[17],q[4];
cp(-pi/16384) q[18],q[4];
cp(-pi/32768) q[19],q[4];
cp(-pi/65536) q[20],q[4];
cp(-pi/131072) q[21],q[4];
cp(-pi/262144) q[22],q[4];
cp(-pi/524288) q[23],q[4];
cp(-pi/1048576) q[24],q[4];
cp(-pi/2097152) q[25],q[4];
cp(-pi/4194304) q[26],q[4];
cp(-pi/8388608) q[27],q[4];
cp(-pi/16777216) q[28],q[4];
cp(-pi/33554432) q[29],q[4];
cp(-pi/67108864) q[30],q[4];
cp(-pi/134217728) q[31],q[4];
cp(-pi/268435456) q[32],q[4];
cp(-pi/536870912) q[33],q[4];
cp(-pi/1073741824) q[34],q[4];
cp(-pi/2147483648) q[35],q[4];
cp(-pi/4294967296) q[36],q[4];
cp(-pi/8589934592) q[37],q[4];
cp(-pi/17179869184) q[38],q[4];
cp(-pi/34359738368) q[39],q[4];
cp(-pi/68719476736) q[40],q[4];
cp(-pi/137438953472) q[41],q[4];
cp(-pi/274877906944) q[42],q[4];
cp(-pi/549755813888) q[43],q[4];
cp(-pi/1099511627776) q[44],q[4];
cp(-pi/2199023255552) q[45],q[4];
cp(0) q[46],q[4];
cp(0) q[47],q[4];
cp(-pi/16) q[9],q[5];
cp(-pi/32) q[10],q[5];
cp(-pi/64) q[11],q[5];
cp(-pi/128) q[12],q[5];
cp(-pi/256) q[13],q[5];
cp(-pi/512) q[14],q[5];
cp(-pi/1024) q[15],q[5];
cp(-pi/2048) q[16],q[5];
cp(-pi/4096) q[17],q[5];
cp(-pi/8192) q[18],q[5];
cp(-pi/16384) q[19],q[5];
cp(-pi/32768) q[20],q[5];
cp(-pi/65536) q[21],q[5];
cp(-pi/131072) q[22],q[5];
cp(-pi/262144) q[23],q[5];
cp(-pi/524288) q[24],q[5];
cp(-pi/1048576) q[25],q[5];
cp(-pi/2097152) q[26],q[5];
cp(-pi/4194304) q[27],q[5];
cp(-pi/8388608) q[28],q[5];
cp(-pi/16777216) q[29],q[5];
cp(-pi/33554432) q[30],q[5];
cp(-pi/67108864) q[31],q[5];
cp(-pi/134217728) q[32],q[5];
cp(-pi/268435456) q[33],q[5];
cp(-pi/536870912) q[34],q[5];
cp(-pi/1073741824) q[35],q[5];
cp(-pi/2147483648) q[36],q[5];
cp(-pi/4294967296) q[37],q[5];
cp(-pi/8589934592) q[38],q[5];
cp(-pi/17179869184) q[39],q[5];
cp(-pi/34359738368) q[40],q[5];
cp(-pi/68719476736) q[41],q[5];
cp(-pi/137438953472) q[42],q[5];
cp(-pi/274877906944) q[43],q[5];
cp(-pi/549755813888) q[44],q[5];
cp(-pi/1099511627776) q[45],q[5];
cp(-pi/2199023255552) q[46],q[5];
cp(0) q[47],q[5];
cp(-pi/8) q[9],q[6];
cp(-pi/16) q[10],q[6];
cp(-pi/32) q[11],q[6];
cp(-pi/64) q[12],q[6];
cp(-pi/128) q[13],q[6];
cp(-pi/256) q[14],q[6];
cp(-pi/512) q[15],q[6];
cp(-pi/1024) q[16],q[6];
cp(-pi/2048) q[17],q[6];
cp(-pi/4096) q[18],q[6];
cp(-pi/8192) q[19],q[6];
cp(-pi/16384) q[20],q[6];
cp(-pi/32768) q[21],q[6];
cp(-pi/65536) q[22],q[6];
cp(-pi/131072) q[23],q[6];
cp(-pi/262144) q[24],q[6];
cp(-pi/524288) q[25],q[6];
cp(-pi/1048576) q[26],q[6];
cp(-pi/2097152) q[27],q[6];
cp(-pi/4194304) q[28],q[6];
cp(-pi/8388608) q[29],q[6];
cp(-pi/16777216) q[30],q[6];
cp(-pi/33554432) q[31],q[6];
cp(-pi/67108864) q[32],q[6];
cp(-pi/134217728) q[33],q[6];
cp(-pi/268435456) q[34],q[6];
cp(-pi/536870912) q[35],q[6];
cp(-pi/1073741824) q[36],q[6];
cp(-pi/2147483648) q[37],q[6];
cp(-pi/4294967296) q[38],q[6];
cp(-pi/8589934592) q[39],q[6];
cp(-pi/17179869184) q[40],q[6];
cp(-pi/34359738368) q[41],q[6];
cp(-pi/68719476736) q[42],q[6];
cp(-pi/137438953472) q[43],q[6];
cp(-pi/274877906944) q[44],q[6];
cp(-pi/549755813888) q[45],q[6];
cp(-pi/1099511627776) q[46],q[6];
cp(-pi/2199023255552) q[47],q[6];
cp(-pi/4) q[9],q[7];
cp(-pi/8) q[10],q[7];
cp(-pi/16) q[11],q[7];
cp(-pi/32) q[12],q[7];
cp(-pi/64) q[13],q[7];
cp(-pi/128) q[14],q[7];
cp(-pi/256) q[15],q[7];
cp(-pi/512) q[16],q[7];
cp(-pi/1024) q[17],q[7];
cp(-pi/2048) q[18],q[7];
cp(-pi/4096) q[19],q[7];
cp(-pi/8192) q[20],q[7];
cp(-pi/16384) q[21],q[7];
cp(-pi/32768) q[22],q[7];
cp(-pi/65536) q[23],q[7];
cp(-pi/131072) q[24],q[7];
cp(-pi/262144) q[25],q[7];
cp(-pi/524288) q[26],q[7];
cp(-pi/1048576) q[27],q[7];
cp(-pi/2097152) q[28],q[7];
cp(-pi/4194304) q[29],q[7];
cp(-pi/8388608) q[30],q[7];
cp(-pi/16777216) q[31],q[7];
cp(-pi/33554432) q[32],q[7];
cp(-pi/67108864) q[33],q[7];
cp(-pi/134217728) q[34],q[7];
cp(-pi/268435456) q[35],q[7];
cp(-pi/536870912) q[36],q[7];
cp(-pi/1073741824) q[37],q[7];
cp(-pi/2147483648) q[38],q[7];
cp(-pi/4294967296) q[39],q[7];
cp(-pi/8589934592) q[40],q[7];
cp(-pi/17179869184) q[41],q[7];
cp(-pi/34359738368) q[42],q[7];
cp(-pi/68719476736) q[43],q[7];
cp(-pi/137438953472) q[44],q[7];
cp(-pi/274877906944) q[45],q[7];
cp(-pi/549755813888) q[46],q[7];
cp(-pi/1099511627776) q[47],q[7];
cp(-pi/2) q[9],q[8];
cp(-pi/4) q[10],q[8];
cp(-pi/8) q[11],q[8];
cp(-pi/16) q[12],q[8];
cp(-pi/32) q[13],q[8];
cp(-pi/64) q[14],q[8];
cp(-pi/128) q[15],q[8];
cp(-pi/256) q[16],q[8];
cp(-pi/512) q[17],q[8];
cp(-pi/1024) q[18],q[8];
cp(-pi/2048) q[19],q[8];
cp(-pi/4096) q[20],q[8];
cp(-pi/8192) q[21],q[8];
cp(-pi/16384) q[22],q[8];
cp(-pi/32768) q[23],q[8];
cp(-pi/65536) q[24],q[8];
cp(-pi/131072) q[25],q[8];
cp(-pi/262144) q[26],q[8];
cp(-pi/524288) q[27],q[8];
cp(-pi/1048576) q[28],q[8];
cp(-pi/2097152) q[29],q[8];
cp(-pi/4194304) q[30],q[8];
cp(-pi/8388608) q[31],q[8];
cp(-pi/16777216) q[32],q[8];
cp(-pi/33554432) q[33],q[8];
cp(-pi/67108864) q[34],q[8];
cp(-pi/134217728) q[35],q[8];
cp(-pi/268435456) q[36],q[8];
cp(-pi/536870912) q[37],q[8];
cp(-pi/1073741824) q[38],q[8];
cp(-pi/2147483648) q[39],q[8];
cp(-pi/4294967296) q[40],q[8];
cp(-pi/8589934592) q[41],q[8];
cp(-pi/17179869184) q[42],q[8];
cp(-pi/34359738368) q[43],q[8];
cp(-pi/68719476736) q[44],q[8];
cp(-pi/137438953472) q[45],q[8];
cp(-pi/274877906944) q[46],q[8];
cp(-pi/549755813888) q[47],q[8];
h q[9];
cp(-pi/2) q[10],q[9];
h q[10];
cp(-pi/4) q[11],q[9];
cp(-pi/2) q[11],q[10];
h q[11];
cp(-pi/8) q[12],q[9];
cp(-pi/4) q[12],q[10];
cp(-pi/2) q[12],q[11];
h q[12];
cp(-pi/16) q[13],q[9];
cp(-pi/8) q[13],q[10];
cp(-pi/4) q[13],q[11];
cp(-pi/2) q[13],q[12];
h q[13];
cp(-pi/32) q[14],q[9];
cp(-pi/16) q[14],q[10];
cp(-pi/8) q[14],q[11];
cp(-pi/4) q[14],q[12];
cp(-pi/2) q[14],q[13];
h q[14];
cp(-pi/64) q[15],q[9];
cp(-pi/32) q[15],q[10];
cp(-pi/16) q[15],q[11];
cp(-pi/8) q[15],q[12];
cp(-pi/4) q[15],q[13];
cp(-pi/2) q[15],q[14];
h q[15];
cp(-pi/128) q[16],q[9];
cp(-pi/64) q[16],q[10];
cp(-pi/32) q[16],q[11];
cp(-pi/16) q[16],q[12];
cp(-pi/8) q[16],q[13];
cp(-pi/4) q[16],q[14];
cp(-pi/2) q[16],q[15];
h q[16];
cp(-pi/256) q[17],q[9];
cp(-pi/128) q[17],q[10];
cp(-pi/64) q[17],q[11];
cp(-pi/32) q[17],q[12];
cp(-pi/16) q[17],q[13];
cp(-pi/8) q[17],q[14];
cp(-pi/4) q[17],q[15];
cp(-pi/2) q[17],q[16];
h q[17];
cp(-pi/512) q[18],q[9];
cp(-pi/256) q[18],q[10];
cp(-pi/128) q[18],q[11];
cp(-pi/64) q[18],q[12];
cp(-pi/32) q[18],q[13];
cp(-pi/16) q[18],q[14];
cp(-pi/8) q[18],q[15];
cp(-pi/4) q[18],q[16];
cp(-pi/2) q[18],q[17];
h q[18];
cp(-pi/1024) q[19],q[9];
cp(-pi/512) q[19],q[10];
cp(-pi/256) q[19],q[11];
cp(-pi/128) q[19],q[12];
cp(-pi/64) q[19],q[13];
cp(-pi/32) q[19],q[14];
cp(-pi/16) q[19],q[15];
cp(-pi/8) q[19],q[16];
cp(-pi/4) q[19],q[17];
cp(-pi/2) q[19],q[18];
h q[19];
cp(-pi/2048) q[20],q[9];
cp(-pi/1024) q[20],q[10];
cp(-pi/512) q[20],q[11];
cp(-pi/256) q[20],q[12];
cp(-pi/128) q[20],q[13];
cp(-pi/64) q[20],q[14];
cp(-pi/32) q[20],q[15];
cp(-pi/16) q[20],q[16];
cp(-pi/8) q[20],q[17];
cp(-pi/4) q[20],q[18];
cp(-pi/2) q[20],q[19];
h q[20];
cp(-pi/4096) q[21],q[9];
cp(-pi/2048) q[21],q[10];
cp(-pi/1024) q[21],q[11];
cp(-pi/512) q[21],q[12];
cp(-pi/256) q[21],q[13];
cp(-pi/128) q[21],q[14];
cp(-pi/64) q[21],q[15];
cp(-pi/32) q[21],q[16];
cp(-pi/16) q[21],q[17];
cp(-pi/8) q[21],q[18];
cp(-pi/4) q[21],q[19];
cp(-pi/2) q[21],q[20];
h q[21];
cp(-pi/8192) q[22],q[9];
cp(-pi/4096) q[22],q[10];
cp(-pi/2048) q[22],q[11];
cp(-pi/1024) q[22],q[12];
cp(-pi/512) q[22],q[13];
cp(-pi/256) q[22],q[14];
cp(-pi/128) q[22],q[15];
cp(-pi/64) q[22],q[16];
cp(-pi/32) q[22],q[17];
cp(-pi/16) q[22],q[18];
cp(-pi/8) q[22],q[19];
cp(-pi/4) q[22],q[20];
cp(-pi/2) q[22],q[21];
h q[22];
cp(-pi/16384) q[23],q[9];
cp(-pi/8192) q[23],q[10];
cp(-pi/4096) q[23],q[11];
cp(-pi/2048) q[23],q[12];
cp(-pi/1024) q[23],q[13];
cp(-pi/512) q[23],q[14];
cp(-pi/256) q[23],q[15];
cp(-pi/128) q[23],q[16];
cp(-pi/64) q[23],q[17];
cp(-pi/32) q[23],q[18];
cp(-pi/16) q[23],q[19];
cp(-pi/8) q[23],q[20];
cp(-pi/4) q[23],q[21];
cp(-pi/2) q[23],q[22];
h q[23];
cp(-pi/32768) q[24],q[9];
cp(-pi/16384) q[24],q[10];
cp(-pi/8192) q[24],q[11];
cp(-pi/4096) q[24],q[12];
cp(-pi/2048) q[24],q[13];
cp(-pi/1024) q[24],q[14];
cp(-pi/512) q[24],q[15];
cp(-pi/256) q[24],q[16];
cp(-pi/128) q[24],q[17];
cp(-pi/64) q[24],q[18];
cp(-pi/32) q[24],q[19];
cp(-pi/16) q[24],q[20];
cp(-pi/8) q[24],q[21];
cp(-pi/4) q[24],q[22];
cp(-pi/2) q[24],q[23];
h q[24];
cp(-pi/65536) q[25],q[9];
cp(-pi/32768) q[25],q[10];
cp(-pi/16384) q[25],q[11];
cp(-pi/8192) q[25],q[12];
cp(-pi/4096) q[25],q[13];
cp(-pi/2048) q[25],q[14];
cp(-pi/1024) q[25],q[15];
cp(-pi/512) q[25],q[16];
cp(-pi/256) q[25],q[17];
cp(-pi/128) q[25],q[18];
cp(-pi/64) q[25],q[19];
cp(-pi/32) q[25],q[20];
cp(-pi/16) q[25],q[21];
cp(-pi/8) q[25],q[22];
cp(-pi/4) q[25],q[23];
cp(-pi/2) q[25],q[24];
h q[25];
cp(-pi/131072) q[26],q[9];
cp(-pi/65536) q[26],q[10];
cp(-pi/32768) q[26],q[11];
cp(-pi/16384) q[26],q[12];
cp(-pi/8192) q[26],q[13];
cp(-pi/4096) q[26],q[14];
cp(-pi/2048) q[26],q[15];
cp(-pi/1024) q[26],q[16];
cp(-pi/512) q[26],q[17];
cp(-pi/256) q[26],q[18];
cp(-pi/128) q[26],q[19];
cp(-pi/64) q[26],q[20];
cp(-pi/32) q[26],q[21];
cp(-pi/16) q[26],q[22];
cp(-pi/8) q[26],q[23];
cp(-pi/4) q[26],q[24];
cp(-pi/2) q[26],q[25];
h q[26];
cp(-pi/262144) q[27],q[9];
cp(-pi/131072) q[27],q[10];
cp(-pi/65536) q[27],q[11];
cp(-pi/32768) q[27],q[12];
cp(-pi/16384) q[27],q[13];
cp(-pi/8192) q[27],q[14];
cp(-pi/4096) q[27],q[15];
cp(-pi/2048) q[27],q[16];
cp(-pi/1024) q[27],q[17];
cp(-pi/512) q[27],q[18];
cp(-pi/256) q[27],q[19];
cp(-pi/128) q[27],q[20];
cp(-pi/64) q[27],q[21];
cp(-pi/32) q[27],q[22];
cp(-pi/16) q[27],q[23];
cp(-pi/8) q[27],q[24];
cp(-pi/4) q[27],q[25];
cp(-pi/2) q[27],q[26];
h q[27];
cp(-pi/524288) q[28],q[9];
cp(-pi/262144) q[28],q[10];
cp(-pi/131072) q[28],q[11];
cp(-pi/65536) q[28],q[12];
cp(-pi/32768) q[28],q[13];
cp(-pi/16384) q[28],q[14];
cp(-pi/8192) q[28],q[15];
cp(-pi/4096) q[28],q[16];
cp(-pi/2048) q[28],q[17];
cp(-pi/1024) q[28],q[18];
cp(-pi/512) q[28],q[19];
cp(-pi/256) q[28],q[20];
cp(-pi/128) q[28],q[21];
cp(-pi/64) q[28],q[22];
cp(-pi/32) q[28],q[23];
cp(-pi/16) q[28],q[24];
cp(-pi/8) q[28],q[25];
cp(-pi/4) q[28],q[26];
cp(-pi/2) q[28],q[27];
h q[28];
cp(-pi/1048576) q[29],q[9];
cp(-pi/524288) q[29],q[10];
cp(-pi/262144) q[29],q[11];
cp(-pi/131072) q[29],q[12];
cp(-pi/65536) q[29],q[13];
cp(-pi/32768) q[29],q[14];
cp(-pi/16384) q[29],q[15];
cp(-pi/8192) q[29],q[16];
cp(-pi/4096) q[29],q[17];
cp(-pi/2048) q[29],q[18];
cp(-pi/1024) q[29],q[19];
cp(-pi/512) q[29],q[20];
cp(-pi/256) q[29],q[21];
cp(-pi/128) q[29],q[22];
cp(-pi/64) q[29],q[23];
cp(-pi/32) q[29],q[24];
cp(-pi/16) q[29],q[25];
cp(-pi/8) q[29],q[26];
cp(-pi/4) q[29],q[27];
cp(-pi/2) q[29],q[28];
h q[29];
cp(-pi/2097152) q[30],q[9];
cp(-pi/1048576) q[30],q[10];
cp(-pi/524288) q[30],q[11];
cp(-pi/262144) q[30],q[12];
cp(-pi/131072) q[30],q[13];
cp(-pi/65536) q[30],q[14];
cp(-pi/32768) q[30],q[15];
cp(-pi/16384) q[30],q[16];
cp(-pi/8192) q[30],q[17];
cp(-pi/4096) q[30],q[18];
cp(-pi/2048) q[30],q[19];
cp(-pi/1024) q[30],q[20];
cp(-pi/512) q[30],q[21];
cp(-pi/256) q[30],q[22];
cp(-pi/128) q[30],q[23];
cp(-pi/64) q[30],q[24];
cp(-pi/32) q[30],q[25];
cp(-pi/16) q[30],q[26];
cp(-pi/8) q[30],q[27];
cp(-pi/4) q[30],q[28];
cp(-pi/2) q[30],q[29];
h q[30];
cp(-pi/4194304) q[31],q[9];
cp(-pi/2097152) q[31],q[10];
cp(-pi/1048576) q[31],q[11];
cp(-pi/524288) q[31],q[12];
cp(-pi/262144) q[31],q[13];
cp(-pi/131072) q[31],q[14];
cp(-pi/65536) q[31],q[15];
cp(-pi/32768) q[31],q[16];
cp(-pi/16384) q[31],q[17];
cp(-pi/8192) q[31],q[18];
cp(-pi/4096) q[31],q[19];
cp(-pi/2048) q[31],q[20];
cp(-pi/1024) q[31],q[21];
cp(-pi/512) q[31],q[22];
cp(-pi/256) q[31],q[23];
cp(-pi/128) q[31],q[24];
cp(-pi/64) q[31],q[25];
cp(-pi/32) q[31],q[26];
cp(-pi/16) q[31],q[27];
cp(-pi/8) q[31],q[28];
cp(-pi/4) q[31],q[29];
cp(-pi/2) q[31],q[30];
h q[31];
cp(-pi/8388608) q[32],q[9];
cp(-pi/4194304) q[32],q[10];
cp(-pi/2097152) q[32],q[11];
cp(-pi/1048576) q[32],q[12];
cp(-pi/524288) q[32],q[13];
cp(-pi/262144) q[32],q[14];
cp(-pi/131072) q[32],q[15];
cp(-pi/65536) q[32],q[16];
cp(-pi/32768) q[32],q[17];
cp(-pi/16384) q[32],q[18];
cp(-pi/8192) q[32],q[19];
cp(-pi/4096) q[32],q[20];
cp(-pi/2048) q[32],q[21];
cp(-pi/1024) q[32],q[22];
cp(-pi/512) q[32],q[23];
cp(-pi/256) q[32],q[24];
cp(-pi/128) q[32],q[25];
cp(-pi/64) q[32],q[26];
cp(-pi/32) q[32],q[27];
cp(-pi/16) q[32],q[28];
cp(-pi/8) q[32],q[29];
cp(-pi/4) q[32],q[30];
cp(-pi/2) q[32],q[31];
h q[32];
cp(-pi/16777216) q[33],q[9];
cp(-pi/8388608) q[33],q[10];
cp(-pi/4194304) q[33],q[11];
cp(-pi/2097152) q[33],q[12];
cp(-pi/1048576) q[33],q[13];
cp(-pi/524288) q[33],q[14];
cp(-pi/262144) q[33],q[15];
cp(-pi/131072) q[33],q[16];
cp(-pi/65536) q[33],q[17];
cp(-pi/32768) q[33],q[18];
cp(-pi/16384) q[33],q[19];
cp(-pi/8192) q[33],q[20];
cp(-pi/4096) q[33],q[21];
cp(-pi/2048) q[33],q[22];
cp(-pi/1024) q[33],q[23];
cp(-pi/512) q[33],q[24];
cp(-pi/256) q[33],q[25];
cp(-pi/128) q[33],q[26];
cp(-pi/64) q[33],q[27];
cp(-pi/32) q[33],q[28];
cp(-pi/16) q[33],q[29];
cp(-pi/8) q[33],q[30];
cp(-pi/4) q[33],q[31];
cp(-pi/2) q[33],q[32];
h q[33];
cp(-pi/33554432) q[34],q[9];
cp(-pi/16777216) q[34],q[10];
cp(-pi/8388608) q[34],q[11];
cp(-pi/4194304) q[34],q[12];
cp(-pi/2097152) q[34],q[13];
cp(-pi/1048576) q[34],q[14];
cp(-pi/524288) q[34],q[15];
cp(-pi/262144) q[34],q[16];
cp(-pi/131072) q[34],q[17];
cp(-pi/65536) q[34],q[18];
cp(-pi/32768) q[34],q[19];
cp(-pi/16384) q[34],q[20];
cp(-pi/8192) q[34],q[21];
cp(-pi/4096) q[34],q[22];
cp(-pi/2048) q[34],q[23];
cp(-pi/1024) q[34],q[24];
cp(-pi/512) q[34],q[25];
cp(-pi/256) q[34],q[26];
cp(-pi/128) q[34],q[27];
cp(-pi/64) q[34],q[28];
cp(-pi/32) q[34],q[29];
cp(-pi/16) q[34],q[30];
cp(-pi/8) q[34],q[31];
cp(-pi/4) q[34],q[32];
cp(-pi/2) q[34],q[33];
h q[34];
cp(-pi/67108864) q[35],q[9];
cp(-pi/33554432) q[35],q[10];
cp(-pi/16777216) q[35],q[11];
cp(-pi/8388608) q[35],q[12];
cp(-pi/4194304) q[35],q[13];
cp(-pi/2097152) q[35],q[14];
cp(-pi/1048576) q[35],q[15];
cp(-pi/524288) q[35],q[16];
cp(-pi/262144) q[35],q[17];
cp(-pi/131072) q[35],q[18];
cp(-pi/65536) q[35],q[19];
cp(-pi/32768) q[35],q[20];
cp(-pi/16384) q[35],q[21];
cp(-pi/8192) q[35],q[22];
cp(-pi/4096) q[35],q[23];
cp(-pi/2048) q[35],q[24];
cp(-pi/1024) q[35],q[25];
cp(-pi/512) q[35],q[26];
cp(-pi/256) q[35],q[27];
cp(-pi/128) q[35],q[28];
cp(-pi/64) q[35],q[29];
cp(-pi/32) q[35],q[30];
cp(-pi/16) q[35],q[31];
cp(-pi/8) q[35],q[32];
cp(-pi/4) q[35],q[33];
cp(-pi/2) q[35],q[34];
h q[35];
cp(-pi/134217728) q[36],q[9];
cp(-pi/67108864) q[36],q[10];
cp(-pi/33554432) q[36],q[11];
cp(-pi/16777216) q[36],q[12];
cp(-pi/8388608) q[36],q[13];
cp(-pi/4194304) q[36],q[14];
cp(-pi/2097152) q[36],q[15];
cp(-pi/1048576) q[36],q[16];
cp(-pi/524288) q[36],q[17];
cp(-pi/262144) q[36],q[18];
cp(-pi/131072) q[36],q[19];
cp(-pi/65536) q[36],q[20];
cp(-pi/32768) q[36],q[21];
cp(-pi/16384) q[36],q[22];
cp(-pi/8192) q[36],q[23];
cp(-pi/4096) q[36],q[24];
cp(-pi/2048) q[36],q[25];
cp(-pi/1024) q[36],q[26];
cp(-pi/512) q[36],q[27];
cp(-pi/256) q[36],q[28];
cp(-pi/128) q[36],q[29];
cp(-pi/64) q[36],q[30];
cp(-pi/32) q[36],q[31];
cp(-pi/16) q[36],q[32];
cp(-pi/8) q[36],q[33];
cp(-pi/4) q[36],q[34];
cp(-pi/2) q[36],q[35];
h q[36];
cp(-pi/268435456) q[37],q[9];
cp(-pi/134217728) q[37],q[10];
cp(-pi/67108864) q[37],q[11];
cp(-pi/33554432) q[37],q[12];
cp(-pi/16777216) q[37],q[13];
cp(-pi/8388608) q[37],q[14];
cp(-pi/4194304) q[37],q[15];
cp(-pi/2097152) q[37],q[16];
cp(-pi/1048576) q[37],q[17];
cp(-pi/524288) q[37],q[18];
cp(-pi/262144) q[37],q[19];
cp(-pi/131072) q[37],q[20];
cp(-pi/65536) q[37],q[21];
cp(-pi/32768) q[37],q[22];
cp(-pi/16384) q[37],q[23];
cp(-pi/8192) q[37],q[24];
cp(-pi/4096) q[37],q[25];
cp(-pi/2048) q[37],q[26];
cp(-pi/1024) q[37],q[27];
cp(-pi/512) q[37],q[28];
cp(-pi/256) q[37],q[29];
cp(-pi/128) q[37],q[30];
cp(-pi/64) q[37],q[31];
cp(-pi/32) q[37],q[32];
cp(-pi/16) q[37],q[33];
cp(-pi/8) q[37],q[34];
cp(-pi/4) q[37],q[35];
cp(-pi/2) q[37],q[36];
h q[37];
cp(-pi/536870912) q[38],q[9];
cp(-pi/268435456) q[38],q[10];
cp(-pi/134217728) q[38],q[11];
cp(-pi/67108864) q[38],q[12];
cp(-pi/33554432) q[38],q[13];
cp(-pi/16777216) q[38],q[14];
cp(-pi/8388608) q[38],q[15];
cp(-pi/4194304) q[38],q[16];
cp(-pi/2097152) q[38],q[17];
cp(-pi/1048576) q[38],q[18];
cp(-pi/524288) q[38],q[19];
cp(-pi/262144) q[38],q[20];
cp(-pi/131072) q[38],q[21];
cp(-pi/65536) q[38],q[22];
cp(-pi/32768) q[38],q[23];
cp(-pi/16384) q[38],q[24];
cp(-pi/8192) q[38],q[25];
cp(-pi/4096) q[38],q[26];
cp(-pi/2048) q[38],q[27];
cp(-pi/1024) q[38],q[28];
cp(-pi/512) q[38],q[29];
cp(-pi/256) q[38],q[30];
cp(-pi/128) q[38],q[31];
cp(-pi/64) q[38],q[32];
cp(-pi/32) q[38],q[33];
cp(-pi/16) q[38],q[34];
cp(-pi/8) q[38],q[35];
cp(-pi/4) q[38],q[36];
cp(-pi/2) q[38],q[37];
h q[38];
cp(-pi/1073741824) q[39],q[9];
cp(-pi/536870912) q[39],q[10];
cp(-pi/268435456) q[39],q[11];
cp(-pi/134217728) q[39],q[12];
cp(-pi/67108864) q[39],q[13];
cp(-pi/33554432) q[39],q[14];
cp(-pi/16777216) q[39],q[15];
cp(-pi/8388608) q[39],q[16];
cp(-pi/4194304) q[39],q[17];
cp(-pi/2097152) q[39],q[18];
cp(-pi/1048576) q[39],q[19];
cp(-pi/524288) q[39],q[20];
cp(-pi/262144) q[39],q[21];
cp(-pi/131072) q[39],q[22];
cp(-pi/65536) q[39],q[23];
cp(-pi/32768) q[39],q[24];
cp(-pi/16384) q[39],q[25];
cp(-pi/8192) q[39],q[26];
cp(-pi/4096) q[39],q[27];
cp(-pi/2048) q[39],q[28];
cp(-pi/1024) q[39],q[29];
cp(-pi/512) q[39],q[30];
cp(-pi/256) q[39],q[31];
cp(-pi/128) q[39],q[32];
cp(-pi/64) q[39],q[33];
cp(-pi/32) q[39],q[34];
cp(-pi/16) q[39],q[35];
cp(-pi/8) q[39],q[36];
cp(-pi/4) q[39],q[37];
cp(-pi/2) q[39],q[38];
h q[39];
cp(-pi/2147483648) q[40],q[9];
cp(-pi/1073741824) q[40],q[10];
cp(-pi/536870912) q[40],q[11];
cp(-pi/268435456) q[40],q[12];
cp(-pi/134217728) q[40],q[13];
cp(-pi/67108864) q[40],q[14];
cp(-pi/33554432) q[40],q[15];
cp(-pi/16777216) q[40],q[16];
cp(-pi/8388608) q[40],q[17];
cp(-pi/4194304) q[40],q[18];
cp(-pi/2097152) q[40],q[19];
cp(-pi/1048576) q[40],q[20];
cp(-pi/524288) q[40],q[21];
cp(-pi/262144) q[40],q[22];
cp(-pi/131072) q[40],q[23];
cp(-pi/65536) q[40],q[24];
cp(-pi/32768) q[40],q[25];
cp(-pi/16384) q[40],q[26];
cp(-pi/8192) q[40],q[27];
cp(-pi/4096) q[40],q[28];
cp(-pi/2048) q[40],q[29];
cp(-pi/1024) q[40],q[30];
cp(-pi/512) q[40],q[31];
cp(-pi/256) q[40],q[32];
cp(-pi/128) q[40],q[33];
cp(-pi/64) q[40],q[34];
cp(-pi/32) q[40],q[35];
cp(-pi/16) q[40],q[36];
cp(-pi/8) q[40],q[37];
cp(-pi/4) q[40],q[38];
cp(-pi/2) q[40],q[39];
h q[40];
cp(-pi/4294967296) q[41],q[9];
cp(-pi/2147483648) q[41],q[10];
cp(-pi/1073741824) q[41],q[11];
cp(-pi/536870912) q[41],q[12];
cp(-pi/268435456) q[41],q[13];
cp(-pi/134217728) q[41],q[14];
cp(-pi/67108864) q[41],q[15];
cp(-pi/33554432) q[41],q[16];
cp(-pi/16777216) q[41],q[17];
cp(-pi/8388608) q[41],q[18];
cp(-pi/4194304) q[41],q[19];
cp(-pi/2097152) q[41],q[20];
cp(-pi/1048576) q[41],q[21];
cp(-pi/524288) q[41],q[22];
cp(-pi/262144) q[41],q[23];
cp(-pi/131072) q[41],q[24];
cp(-pi/65536) q[41],q[25];
cp(-pi/32768) q[41],q[26];
cp(-pi/16384) q[41],q[27];
cp(-pi/8192) q[41],q[28];
cp(-pi/4096) q[41],q[29];
cp(-pi/2048) q[41],q[30];
cp(-pi/1024) q[41],q[31];
cp(-pi/512) q[41],q[32];
cp(-pi/256) q[41],q[33];
cp(-pi/128) q[41],q[34];
cp(-pi/64) q[41],q[35];
cp(-pi/32) q[41],q[36];
cp(-pi/16) q[41],q[37];
cp(-pi/8) q[41],q[38];
cp(-pi/4) q[41],q[39];
cp(-pi/2) q[41],q[40];
h q[41];
cp(-pi/8589934592) q[42],q[9];
cp(-pi/4294967296) q[42],q[10];
cp(-pi/2147483648) q[42],q[11];
cp(-pi/1073741824) q[42],q[12];
cp(-pi/536870912) q[42],q[13];
cp(-pi/268435456) q[42],q[14];
cp(-pi/134217728) q[42],q[15];
cp(-pi/67108864) q[42],q[16];
cp(-pi/33554432) q[42],q[17];
cp(-pi/16777216) q[42],q[18];
cp(-pi/8388608) q[42],q[19];
cp(-pi/4194304) q[42],q[20];
cp(-pi/2097152) q[42],q[21];
cp(-pi/1048576) q[42],q[22];
cp(-pi/524288) q[42],q[23];
cp(-pi/262144) q[42],q[24];
cp(-pi/131072) q[42],q[25];
cp(-pi/65536) q[42],q[26];
cp(-pi/32768) q[42],q[27];
cp(-pi/16384) q[42],q[28];
cp(-pi/8192) q[42],q[29];
cp(-pi/4096) q[42],q[30];
cp(-pi/2048) q[42],q[31];
cp(-pi/1024) q[42],q[32];
cp(-pi/512) q[42],q[33];
cp(-pi/256) q[42],q[34];
cp(-pi/128) q[42],q[35];
cp(-pi/64) q[42],q[36];
cp(-pi/32) q[42],q[37];
cp(-pi/16) q[42],q[38];
cp(-pi/8) q[42],q[39];
cp(-pi/4) q[42],q[40];
cp(-pi/2) q[42],q[41];
h q[42];
cp(-pi/17179869184) q[43],q[9];
cp(-pi/8589934592) q[43],q[10];
cp(-pi/4294967296) q[43],q[11];
cp(-pi/2147483648) q[43],q[12];
cp(-pi/1073741824) q[43],q[13];
cp(-pi/536870912) q[43],q[14];
cp(-pi/268435456) q[43],q[15];
cp(-pi/134217728) q[43],q[16];
cp(-pi/67108864) q[43],q[17];
cp(-pi/33554432) q[43],q[18];
cp(-pi/16777216) q[43],q[19];
cp(-pi/8388608) q[43],q[20];
cp(-pi/4194304) q[43],q[21];
cp(-pi/2097152) q[43],q[22];
cp(-pi/1048576) q[43],q[23];
cp(-pi/524288) q[43],q[24];
cp(-pi/262144) q[43],q[25];
cp(-pi/131072) q[43],q[26];
cp(-pi/65536) q[43],q[27];
cp(-pi/32768) q[43],q[28];
cp(-pi/16384) q[43],q[29];
cp(-pi/8192) q[43],q[30];
cp(-pi/4096) q[43],q[31];
cp(-pi/2048) q[43],q[32];
cp(-pi/1024) q[43],q[33];
cp(-pi/512) q[43],q[34];
cp(-pi/256) q[43],q[35];
cp(-pi/128) q[43],q[36];
cp(-pi/64) q[43],q[37];
cp(-pi/32) q[43],q[38];
cp(-pi/16) q[43],q[39];
cp(-pi/8) q[43],q[40];
cp(-pi/4) q[43],q[41];
cp(-pi/2) q[43],q[42];
h q[43];
cp(-pi/34359738368) q[44],q[9];
cp(-pi/17179869184) q[44],q[10];
cp(-pi/8589934592) q[44],q[11];
cp(-pi/4294967296) q[44],q[12];
cp(-pi/2147483648) q[44],q[13];
cp(-pi/1073741824) q[44],q[14];
cp(-pi/536870912) q[44],q[15];
cp(-pi/268435456) q[44],q[16];
cp(-pi/134217728) q[44],q[17];
cp(-pi/67108864) q[44],q[18];
cp(-pi/33554432) q[44],q[19];
cp(-pi/16777216) q[44],q[20];
cp(-pi/8388608) q[44],q[21];
cp(-pi/4194304) q[44],q[22];
cp(-pi/2097152) q[44],q[23];
cp(-pi/1048576) q[44],q[24];
cp(-pi/524288) q[44],q[25];
cp(-pi/262144) q[44],q[26];
cp(-pi/131072) q[44],q[27];
cp(-pi/65536) q[44],q[28];
cp(-pi/32768) q[44],q[29];
cp(-pi/16384) q[44],q[30];
cp(-pi/8192) q[44],q[31];
cp(-pi/4096) q[44],q[32];
cp(-pi/2048) q[44],q[33];
cp(-pi/1024) q[44],q[34];
cp(-pi/512) q[44],q[35];
cp(-pi/256) q[44],q[36];
cp(-pi/128) q[44],q[37];
cp(-pi/64) q[44],q[38];
cp(-pi/32) q[44],q[39];
cp(-pi/16) q[44],q[40];
cp(-pi/8) q[44],q[41];
cp(-pi/4) q[44],q[42];
cp(-pi/2) q[44],q[43];
h q[44];
cp(-pi/68719476736) q[45],q[9];
cp(-pi/34359738368) q[45],q[10];
cp(-pi/17179869184) q[45],q[11];
cp(-pi/8589934592) q[45],q[12];
cp(-pi/4294967296) q[45],q[13];
cp(-pi/2147483648) q[45],q[14];
cp(-pi/1073741824) q[45],q[15];
cp(-pi/536870912) q[45],q[16];
cp(-pi/268435456) q[45],q[17];
cp(-pi/134217728) q[45],q[18];
cp(-pi/67108864) q[45],q[19];
cp(-pi/33554432) q[45],q[20];
cp(-pi/16777216) q[45],q[21];
cp(-pi/8388608) q[45],q[22];
cp(-pi/4194304) q[45],q[23];
cp(-pi/2097152) q[45],q[24];
cp(-pi/1048576) q[45],q[25];
cp(-pi/524288) q[45],q[26];
cp(-pi/262144) q[45],q[27];
cp(-pi/131072) q[45],q[28];
cp(-pi/65536) q[45],q[29];
cp(-pi/32768) q[45],q[30];
cp(-pi/16384) q[45],q[31];
cp(-pi/8192) q[45],q[32];
cp(-pi/4096) q[45],q[33];
cp(-pi/2048) q[45],q[34];
cp(-pi/1024) q[45],q[35];
cp(-pi/512) q[45],q[36];
cp(-pi/256) q[45],q[37];
cp(-pi/128) q[45],q[38];
cp(-pi/64) q[45],q[39];
cp(-pi/32) q[45],q[40];
cp(-pi/16) q[45],q[41];
cp(-pi/8) q[45],q[42];
cp(-pi/4) q[45],q[43];
cp(-pi/2) q[45],q[44];
h q[45];
cp(-pi/137438953472) q[46],q[9];
cp(-pi/68719476736) q[46],q[10];
cp(-pi/34359738368) q[46],q[11];
cp(-pi/17179869184) q[46],q[12];
cp(-pi/8589934592) q[46],q[13];
cp(-pi/4294967296) q[46],q[14];
cp(-pi/2147483648) q[46],q[15];
cp(-pi/1073741824) q[46],q[16];
cp(-pi/536870912) q[46],q[17];
cp(-pi/268435456) q[46],q[18];
cp(-pi/134217728) q[46],q[19];
cp(-pi/67108864) q[46],q[20];
cp(-pi/33554432) q[46],q[21];
cp(-pi/16777216) q[46],q[22];
cp(-pi/8388608) q[46],q[23];
cp(-pi/4194304) q[46],q[24];
cp(-pi/2097152) q[46],q[25];
cp(-pi/1048576) q[46],q[26];
cp(-pi/524288) q[46],q[27];
cp(-pi/262144) q[46],q[28];
cp(-pi/131072) q[46],q[29];
cp(-pi/65536) q[46],q[30];
cp(-pi/32768) q[46],q[31];
cp(-pi/16384) q[46],q[32];
cp(-pi/8192) q[46],q[33];
cp(-pi/4096) q[46],q[34];
cp(-pi/2048) q[46],q[35];
cp(-pi/1024) q[46],q[36];
cp(-pi/512) q[46],q[37];
cp(-pi/256) q[46],q[38];
cp(-pi/128) q[46],q[39];
cp(-pi/64) q[46],q[40];
cp(-pi/32) q[46],q[41];
cp(-pi/16) q[46],q[42];
cp(-pi/8) q[46],q[43];
cp(-pi/4) q[46],q[44];
cp(-pi/2) q[46],q[45];
h q[46];
cp(-pi/274877906944) q[47],q[9];
cp(-pi/137438953472) q[47],q[10];
cp(-pi/68719476736) q[47],q[11];
cp(-pi/34359738368) q[47],q[12];
cp(-pi/17179869184) q[47],q[13];
cp(-pi/8589934592) q[47],q[14];
cp(-pi/4294967296) q[47],q[15];
cp(-pi/2147483648) q[47],q[16];
cp(-pi/1073741824) q[47],q[17];
cp(-pi/536870912) q[47],q[18];
cp(-pi/268435456) q[47],q[19];
cp(-pi/134217728) q[47],q[20];
cp(-pi/67108864) q[47],q[21];
cp(-pi/33554432) q[47],q[22];
cp(-pi/16777216) q[47],q[23];
cp(-pi/8388608) q[47],q[24];
cp(-pi/4194304) q[47],q[25];
cp(-pi/2097152) q[47],q[26];
cp(-pi/1048576) q[47],q[27];
cp(-pi/524288) q[47],q[28];
cp(-pi/262144) q[47],q[29];
cp(-pi/131072) q[47],q[30];
cp(-pi/65536) q[47],q[31];
cp(-pi/32768) q[47],q[32];
cp(-pi/16384) q[47],q[33];
cp(-pi/8192) q[47],q[34];
cp(-pi/4096) q[47],q[35];
cp(-pi/2048) q[47],q[36];
cp(-pi/1024) q[47],q[37];
cp(-pi/512) q[47],q[38];
cp(-pi/256) q[47],q[39];
cp(-pi/128) q[47],q[40];
cp(-pi/64) q[47],q[41];
cp(-pi/32) q[47],q[42];
cp(-pi/16) q[47],q[43];
cp(-pi/8) q[47],q[44];
cp(-pi/4) q[47],q[45];
cp(-pi/2) q[47],q[46];
h q[47];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26],q[27],q[28],q[29],q[30],q[31],q[32],q[33],q[34],q[35],q[36],q[37],q[38],q[39],q[40],q[41],q[42],q[43],q[44],q[45],q[46],q[47],psi[0];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
c[4] = measure q[4];
c[5] = measure q[5];
c[6] = measure q[6];
c[7] = measure q[7];
c[8] = measure q[8];
c[9] = measure q[9];
c[10] = measure q[10];
c[11] = measure q[11];
c[12] = measure q[12];
c[13] = measure q[13];
c[14] = measure q[14];
c[15] = measure q[15];
c[16] = measure q[16];
c[17] = measure q[17];
c[18] = measure q[18];
c[19] = measure q[19];
c[20] = measure q[20];
c[21] = measure q[21];
c[22] = measure q[22];
c[23] = measure q[23];
c[24] = measure q[24];
c[25] = measure q[25];
c[26] = measure q[26];
c[27] = measure q[27];
c[28] = measure q[28];
c[29] = measure q[29];
c[30] = measure q[30];
c[31] = measure q[31];
c[32] = measure q[32];
c[33] = measure q[33];
c[34] = measure q[34];
c[35] = measure q[35];
c[36] = measure q[36];
c[37] = measure q[37];
c[38] = measure q[38];
c[39] = measure q[39];
c[40] = measure q[40];
c[41] = measure q[41];
c[42] = measure q[42];
c[43] = measure q[43];
c[44] = measure q[44];
c[45] = measure q[45];
c[46] = measure q[46];
c[47] = measure q[47];