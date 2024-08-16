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

qubit[14] q;
bit[14] meas;
ry(1.11329148295581) q[0];
ry(-0.579989909202188) q[1];
cz q[0],q[1];
ry(-5.19545259464085) q[2];
cz q[0],q[2];
cz q[1],q[2];
ry(-2.27463485453992) q[3];
cz q[0],q[3];
cz q[1],q[3];
cz q[2],q[3];
ry(2.21034684200592) q[4];
cz q[0],q[4];
cz q[1],q[4];
cz q[2],q[4];
cz q[3],q[4];
ry(0.0810684405836746) q[5];
cz q[0],q[5];
cz q[1],q[5];
cz q[2],q[5];
cz q[3],q[5];
cz q[4],q[5];
ry(2.32280424510639) q[6];
cz q[0],q[6];
cz q[1],q[6];
cz q[2],q[6];
cz q[3],q[6];
cz q[4],q[6];
cz q[5],q[6];
ry(2.90984633968069) q[7];
cz q[0],q[7];
cz q[1],q[7];
cz q[2],q[7];
cz q[3],q[7];
cz q[4],q[7];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.68824019380042) q[8];
cz q[0],q[8];
cz q[1],q[8];
cz q[2],q[8];
cz q[3],q[8];
cz q[4],q[8];
cz q[5],q[8];
cz q[6],q[8];
cz q[7],q[8];
ry(4.31529987636185) q[9];
cz q[0],q[9];
cz q[1],q[9];
cz q[2],q[9];
cz q[3],q[9];
cz q[4],q[9];
cz q[5],q[9];
cz q[6],q[9];
cz q[7],q[9];
cz q[8],q[9];
ry(-4.92746542736229) q[10];
cz q[0],q[10];
cz q[1],q[10];
cz q[2],q[10];
cz q[3],q[10];
cz q[4],q[10];
cz q[5],q[10];
cz q[6],q[10];
cz q[7],q[10];
cz q[8],q[10];
cz q[9],q[10];
ry(4.96740692930811) q[11];
cz q[0],q[11];
cz q[1],q[11];
cz q[2],q[11];
cz q[3],q[11];
cz q[4],q[11];
cz q[5],q[11];
cz q[6],q[11];
cz q[7],q[11];
cz q[8],q[11];
cz q[9],q[11];
cz q[10],q[11];
ry(4.30486769699809) q[12];
cz q[0],q[12];
cz q[1],q[12];
cz q[2],q[12];
cz q[3],q[12];
cz q[4],q[12];
cz q[5],q[12];
cz q[6],q[12];
cz q[7],q[12];
cz q[8],q[12];
cz q[9],q[12];
cz q[10],q[12];
cz q[11],q[12];
ry(-3.66943545549979) q[13];
cz q[0],q[13];
ry(-0.0204788950699362) q[0];
cz q[1],q[13];
ry(3.85639596330695) q[1];
cz q[0],q[1];
cz q[2],q[13];
ry(3.8983736881617) q[2];
cz q[0],q[2];
cz q[1],q[2];
cz q[3],q[13];
ry(-2.0822399268384) q[3];
cz q[0],q[3];
cz q[1],q[3];
cz q[2],q[3];
cz q[4],q[13];
ry(-1.74906044826772) q[4];
cz q[0],q[4];
cz q[1],q[4];
cz q[2],q[4];
cz q[3],q[4];
cz q[5],q[13];
ry(-0.820049942086035) q[5];
cz q[0],q[5];
cz q[1],q[5];
cz q[2],q[5];
cz q[3],q[5];
cz q[4],q[5];
cz q[6],q[13];
ry(-1.05791754068145) q[6];
cz q[0],q[6];
cz q[1],q[6];
cz q[2],q[6];
cz q[3],q[6];
cz q[4],q[6];
cz q[5],q[6];
cz q[7],q[13];
ry(5.81975879592245) q[7];
cz q[0],q[7];
cz q[1],q[7];
cz q[2],q[7];
cz q[3],q[7];
cz q[4],q[7];
cz q[5],q[7];
cz q[6],q[7];
cz q[8],q[13];
ry(3.02780619095389) q[8];
cz q[0],q[8];
cz q[1],q[8];
cz q[2],q[8];
cz q[3],q[8];
cz q[4],q[8];
cz q[5],q[8];
cz q[6],q[8];
cz q[7],q[8];
cz q[9],q[13];
cz q[10],q[13];
ry(-6.13429981891083) q[10];
cz q[11],q[13];
ry(-5.16986632767649) q[11];
cz q[12],q[13];
ry(-2.62443972791364) q[12];
ry(0.702613579727363) q[13];
ry(-4.83733546838638) q[9];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
ry(-2.44581096025773) q[0];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
ry(5.9268883796862) q[1];
cz q[0],q[1];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
ry(0.223531299184564) q[2];
cz q[0],q[2];
cz q[1],q[2];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
ry(5.84415864311613) q[3];
cz q[0],q[3];
cz q[1],q[3];
cz q[2],q[3];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
ry(0.303748076404419) q[4];
cz q[0],q[4];
cz q[1],q[4];
cz q[2],q[4];
cz q[3],q[4];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
ry(-4.86671577111088) q[5];
cz q[0],q[5];
cz q[1],q[5];
cz q[2],q[5];
cz q[3],q[5];
cz q[4],q[5];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
ry(-1.30266470661662) q[6];
cz q[0],q[6];
cz q[1],q[6];
cz q[2],q[6];
cz q[3],q[6];
cz q[4],q[6];
cz q[5],q[6];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
ry(-5.32947124273677) q[7];
cz q[0],q[7];
cz q[1],q[7];
cz q[2],q[7];
cz q[3],q[7];
cz q[4],q[7];
cz q[5],q[7];
cz q[6],q[7];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
ry(1.39072524784524) q[8];
cz q[0],q[8];
cz q[1],q[8];
cz q[2],q[8];
cz q[3],q[8];
cz q[4],q[8];
cz q[5],q[8];
cz q[6],q[8];
cz q[7],q[8];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
cz q[9],q[12];
cz q[10],q[12];
cz q[11],q[12];
cz q[9],q[13];
cz q[10],q[13];
ry(3.15448056900281) q[10];
cz q[11],q[13];
ry(5.69367977326462) q[11];
cz q[12],q[13];
ry(5.26567754957095) q[12];
ry(0.893938856947673) q[13];
ry(1.42559846836502) q[9];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
ry(-4.45316550606332) q[0];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
ry(4.36153058886799) q[1];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
ry(-4.31692776621078) q[2];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
ry(-4.6838188836876) q[3];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
ry(-3.76137354874322) q[4];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
ry(-0.18279824523468) q[5];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
ry(5.47744697412037) q[6];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
ry(3.61611607022444) q[7];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
ry(3.24198001272168) q[8];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
cz q[9],q[12];
cz q[10],q[12];
cz q[11],q[12];
cz q[9],q[13];
cz q[10],q[13];
ry(-1.96684052159203) q[10];
cz q[11],q[13];
ry(-5.16451334100513) q[11];
cz q[12],q[13];
ry(-4.66741993833124) q[12];
ry(-5.48243280597148) q[13];
ry(-4.65587521818598) q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13];
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
meas[3] = measure q[3];
meas[4] = measure q[4];
meas[5] = measure q[5];
meas[6] = measure q[6];
meas[7] = measure q[7];
meas[8] = measure q[8];
meas[9] = measure q[9];
meas[10] = measure q[10];
meas[11] = measure q[11];
meas[12] = measure q[12];
meas[13] = measure q[13];