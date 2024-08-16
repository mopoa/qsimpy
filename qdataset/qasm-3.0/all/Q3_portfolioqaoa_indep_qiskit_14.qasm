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
gate rzz(theta) a,b
{
  cx a,b;
  u3(0,0,theta) b;
  cx a,b;
}

// relative-phase CCX

// relative-phase 3-controlled X gate

// 3-controlled X gate

// 3-controlled sqrt(X) gate, this equals the C3X gate 
// where the CU1 rotations are -pi/8 not -pi/4

// 4-controlled X gate 

qubit[14] q;
bit[14] meas;
bit[14] meas11;
u2(0.031063286073218954,-pi) q[0];
u2(0.022227972955836783,-pi) q[1];
rzz(5.296670482746914) q[0],q[1];
u2(0.004399623827332277,-pi) q[2];
rzz(5.296360987581752) q[0],q[2];
rzz(5.296368789125708) q[1],q[2];
u2(0.0008139994502185388,-pi) q[3];
rzz(5.2965656864072015) q[0],q[3];
rzz(5.2967130886253155) q[1],q[3];
rzz(5.296407427000809) q[2],q[3];
u2(0.022929395978838585,-pi) q[4];
rzz(5.2963709509277495) q[0],q[4];
rzz(5.296511618566411) q[1],q[4];
rzz(5.296389772039776) q[2],q[4];
rzz(5.2965702623549475) q[3],q[4];
u2(0.004916200482836697,-pi) q[5];
rzz(5.296407017467384) q[0],q[5];
rzz(5.296009485613254) q[1],q[5];
rzz(5.2963927540096485) q[2],q[5];
rzz(5.296562105162882) q[3],q[5];
rzz(5.296258576254721) q[4],q[5];
u2(-0.021078228025040335,-pi) q[6];
rzz(5.29624811449512) q[0],q[6];
rzz(5.29634754516426) q[1],q[6];
rzz(5.296463844140825) q[2],q[6];
rzz(5.296535019198953) q[3],q[6];
rzz(5.296491024348712) q[4],q[6];
rzz(5.296213168345223) q[5],q[6];
u2(-0.011568509380235792,-pi) q[7];
rzz(5.296100632155527) q[0],q[7];
rzz(5.296654951690284) q[1],q[7];
rzz(5.296239403480809) q[2],q[7];
rzz(5.296485703576104) q[3],q[7];
rzz(5.296450686464152) q[4],q[7];
rzz(5.296589983451177) q[5],q[7];
rzz(5.296396784099901) q[6],q[7];
u2(0.005296107957038387,-pi) q[8];
rzz(5.296684189415196) q[0],q[8];
rzz(5.296233263591911) q[1],q[8];
rzz(5.296224432995686) q[2],q[8];
rzz(5.2962549048983645) q[3],q[8];
rzz(5.2963888651652145) q[4],q[8];
rzz(5.296372293196005) q[5],q[8];
rzz(5.296212431795712) q[6],q[8];
rzz(5.296381667146072) q[7],q[8];
u2(-0.0022141345670441126,-pi) q[9];
rzz(5.29652624208412) q[0],q[9];
rzz(5.29619192627428) q[1],q[9];
rzz(5.29650082393325) q[2],q[9];
rzz(5.296330557760123) q[3],q[9];
rzz(5.296401029699646) q[4],q[9];
rzz(5.296185952591318) q[5],q[9];
rzz(5.296504962315095) q[6],q[9];
rzz(5.29634340366675) q[7],q[9];
rzz(5.296658308386882) q[8],q[9];
u2(0.010573647558307364,-pi) q[10];
rzz(5.29659729221665) q[0],q[10];
rzz(5.296233876968012) q[1],q[10];
rzz(5.296366007535433) q[2],q[10];
rzz(5.296283222506303) q[3],q[10];
rzz(5.296374593231657) q[4],q[10];
rzz(5.296333317282599) q[5],q[10];
rzz(5.2964688923055485) q[6],q[10];
rzz(5.296198173530949) q[7],q[10];
rzz(5.296317767664365) q[8],q[10];
rzz(5.296444030698941) q[9],q[10];
u2(0.023185760003821887,-pi) q[11];
rzz(5.29653738056587) q[0],q[11];
rzz(5.296847242251269) q[1],q[11];
rzz(5.296374887480196) q[2],q[11];
rzz(5.296326580796178) q[3],q[11];
rzz(5.296423719174869) q[4],q[11];
rzz(5.296340017361729) q[5],q[11];
rzz(5.296716092060732) q[6],q[11];
rzz(5.296415115239099) q[7],q[11];
rzz(5.296030856067153) q[8],q[11];
rzz(5.296394208743351) q[9],q[11];
rzz(5.296611242629908) q[10],q[11];
u2(-0.009008732534574904,-pi) q[12];
rzz(5.296520265242911) q[0],q[12];
rzz(5.2964787216893825) q[1],q[12];
rzz(5.296455595792493) q[2],q[12];
rzz(5.296401111163395) q[3],q[12];
rzz(5.296430099950466) q[4],q[12];
rzz(5.295948961292532) q[5],q[12];
rzz(5.296327518955655) q[6],q[12];
rzz(5.296354522100987) q[7],q[12];
rzz(5.296519149250533) q[8],q[12];
rzz(5.296408951652922) q[9],q[12];
rzz(5.296308014221615) q[10],q[12];
rzz(5.296309914385589) q[11],q[12];
u2(-0.003188335762385641,-pi) q[13];
rzz(5.296170793788447) q[0],q[13];
u3(2.2364992960695558,1.5867262027808584,-pi/2) q[0];
rzz(5.296428882938306) q[1],q[13];
u3(2.2364992960695558,1.5821952770466883,-pi/2) q[1];
rzz(2.716238833514971) q[0],q[1];
rzz(5.296541435117425) q[2],q[13];
u3(2.2364992960695558,1.5730525421157449,-pi/2) q[2];
rzz(2.716080118188213) q[0],q[2];
rzz(2.716084118976392) q[1],q[2];
rzz(5.296292806406419) q[3],q[13];
u3(2.2364992960695558,1.571213762035459,-pi/2) q[3];
rzz(2.716185091850568) q[0],q[3];
rzz(2.716260682663757) q[1],q[3];
rzz(2.7161039332535184) q[2],q[3];
rzz(5.296517759441269) q[4],q[13];
u3(2.2364992960695558,1.5825549808379673,-pi/2) q[4];
rzz(2.7160852275918272) q[0],q[4];
rzz(2.7161573647776303) q[1],q[4];
rzz(2.7160948794354796) q[2],q[4];
rzz(2.7161874384883298) q[3],q[4];
rzz(5.296268081972791) q[5],q[13];
u3(2.2364992960695558,1.5733174529831917,-pi/2) q[5];
rzz(2.716103723236566) q[0],q[5];
rzz(2.715899860930375) q[1],q[5];
rzz(2.7160964086494253) q[2],q[5];
rzz(2.7161832553166687) q[3],q[5];
rzz(2.7160275996061376) q[4],q[5];
rzz(5.296316974343223) q[6],q[13];
u3(2.2364992960695558,1.5599869887985474,-pi/2) q[6];
rzz(2.716022234606033) q[0],q[6];
rzz(2.7160732246469728) q[1],q[6];
rzz(2.716132865094207) q[2],q[6];
rzz(2.7161693650912326) q[3],q[6];
rzz(2.7161468036498344) q[4],q[6];
rzz(2.7160043135196177) q[5],q[6];
rzz(5.296504376149105) q[7],q[13];
u3(2.2364992960695558,1.5648637632105027,-pi/2) q[7];
rzz(2.7159466027049426) q[0],q[7];
rzz(2.7162308688776986) q[1],q[7];
rzz(2.7160177674232187) q[2],q[7];
rzz(2.7161440750509476) q[3],q[7];
rzz(2.7161261175737685) q[4],q[7];
rzz(2.716197551861928) q[5],q[7];
rzz(2.716098475360519) q[6],q[7];
rzz(5.29620784866652) q[8],q[13];
u3(2.2364992960695558,1.5735122771539487,-pi/2) q[8];
rzz(2.716245862569315) q[0],q[8];
rzz(2.7160146187650738) q[1],q[8];
rzz(2.7160100902582562) q[2],q[8];
rzz(2.7160257168609787) q[3],q[8];
rzz(2.7160944143720194) q[4],q[8];
rzz(2.7160859159339044) q[5],q[8];
rzz(2.716003935802259) q[6],q[8];
rzz(2.7160907230834668) q[7],q[8];
rzz(5.296441651432738) q[9],q[13];
rzz(5.296262824453061) q[10],q[13];
u3(2.2364992960695558,1.5762187052878254,-pi/2) q[10];
rzz(5.296396462238256) q[11],q[13];
u3(2.2364992960695558,1.5826864494506445,-pi/2) q[11];
rzz(5.296408792258257) q[12],q[13];
u3(2.2364992960695558,1.5661764681061463,-pi/2) q[12];
u3(2.2364992960695558,1.5691612842827585,-pi/2) q[13];
u3(2.2364992960695558,1.5696608741906193,-pi/2) q[9];
rzz(2.716164864010745) q[0],q[9];
rzz(2.716201299943419) q[0],q[10];
rzz(2.7161705760475385) q[0],q[11];
rzz(2.716161798966725) q[0],q[12];
rzz(2.7159825829971993) q[0],q[13];
u3(0.41712453960694135,-1.5746181940781687,pi/2) q[0];
rzz(2.7159934201596876) q[1],q[9];
rzz(2.716014933316638) q[1],q[10];
rzz(2.7163294793332233) q[1],q[11];
rzz(2.716140494599435) q[1],q[12];
rzz(2.716114936288506) q[1],q[13];
u3(0.41712453960694135,-1.573531142500264,pi/2) q[1];
rzz(-0.6516751505481319) q[0],q[1];
rzz(2.7161518290732887) q[2],q[9];
rzz(2.7160826925211214) q[2],q[10];
rzz(2.7160872463348587) q[2],q[11];
rzz(2.7161286351757843) q[2],q[12];
rzz(2.7161726553028784) q[2],q[13];
u3(0.41712453960694135,-1.5713376339291365,pi/2) q[2];
rzz(-0.6516370718515226) q[0],q[2];
rzz(-0.6516380317134489) q[1],q[2];
rzz(2.716064513165496) q[3],q[9];
rzz(2.716040238698161) q[3],q[10];
rzz(2.716062473698638) q[3],q[11];
rzz(2.716100694365446) q[3],q[12];
rzz(2.716045153514319) q[3],q[13];
u3(0.4171245396069413,-1.570896477109324,pi/2) q[3];
rzz(-0.6516622569443704) q[0],q[3];
rzz(-0.6516803925567317) q[1],q[3];
rzz(-0.6516427855192889) q[2],q[3];
rzz(2.7161006525892017) q[4],q[9];
rzz(2.716087095438307) q[4],q[10];
rzz(2.7161122882070705) q[4],q[11];
rzz(2.716115560396745) q[4],q[12];
rzz(2.7161605139413956) q[4],q[13];
u3(0.41712453960694135,-1.5736174419888957,pi/2) q[4];
rzz(-0.6516382976904763) q[0],q[4];
rzz(-0.6516556047147477) q[1],q[4];
rzz(-0.6516406133434991) q[2],q[4];
rzz(-0.6516628199454946) q[3],q[4];
rzz(2.715990356735279) q[5],q[9];
rzz(2.7160659283039563) q[5],q[10];
rzz(2.7160693642388853) q[5],q[11];
rzz(2.715868822845062) q[5],q[12];
rzz(2.716032474328974) q[5],q[13];
u3(0.41712453960694135,-1.5714011908694525,pi/2) q[5];
rzz(-0.6516427351323982) q[0],q[5];
rzz(-0.6515938248534349) q[1],q[5];
rzz(-0.6516409802297669) q[2],q[5];
rzz(-0.6516618163264504) q[3],q[5];
rzz(-0.6516244716874804) q[4],q[5];
rzz(2.7161539513184825) q[6],q[9];
rzz(2.716135453894304) q[6],q[10];
rzz(2.7162622228856645) q[6],q[11];
rzz(2.7160629548056305) q[6],q[12];
rzz(2.716057547316932) q[6],q[13];
u3(0.4171245396069414,-1.568202969804642,pi/2) q[6];
rzz(-0.6516231845262751) q[0],q[6];
rzz(-0.6516354179654689) q[1],q[6];
rzz(-0.6516497267946302) q[2],q[6];
rzz(-0.6516584838084698) q[3],q[6];
rzz(-0.6516530709079037) q[4],q[6];
rzz(-0.651618884931356) q[5],q[6];
rzz(2.7160711008040033) q[7],q[9];
rzz(2.715996623878179) q[7],q[10];
rzz(2.716107875936279) q[7],q[11];
rzz(2.7160768025600293) q[7],q[12];
rzz(2.7161536507208095) q[7],q[13];
u3(0.41712453960694135,-1.5693729967771923,pi/2) q[7];
rzz(-0.6516050390561782) q[0],q[7];
rzz(-0.6516732396866386) q[1],q[7];
rzz(-0.6516221127677841) q[2],q[7];
rzz(-0.6516524162673514) q[3],q[7];
rzz(-0.6516481079416254) q[4],q[7];
rzz(-0.6516652463279544) q[5],q[7];
rzz(-0.6516414760713869) q[6],q[7];
rzz(2.716232590259023) q[8],q[9];
rzz(2.7160579541479217) q[8],q[10];
rzz(2.7159108201276587) q[8],q[11];
rzz(2.7161612266634547) q[8],q[12];
rzz(2.716001585481708) q[8],q[13];
u3(0.4171245396069413,-1.5714479327351683,pi/2) q[8];
rzz(-0.6516768369462465) q[0],q[8];
rzz(-0.6516213573473678) q[1],q[8];
rzz(-0.6516202708761321) q[2],q[8];
rzz(-0.6516240199826377) q[3],q[8];
rzz(-0.6516405017663076) q[4],q[8];
rzz(-0.6516384628362732) q[5],q[8];
rzz(-0.6516187943100845) q[6],q[8];
rzz(-0.6516396161589768) q[7],q[8];
rzz(2.716122704363906) q[9],q[10];
rzz(2.7160971546660386) q[9],q[11];
rzz(2.7162084539950744) q[10],q[11];
rzz(2.7161047151256947) q[9],q[12];
rzz(2.7160529523869013) q[10],q[12];
rzz(2.71605392682909) q[11],q[12];
rzz(2.716121484228487) q[9],q[13];
rzz(2.716029778167432) q[10],q[13];
u3(0.4171245396069413,-1.5720972541202505,pi/2) q[10];
rzz(2.7160983103034133) q[11],q[13];
u3(0.4171245396069414,-1.5736489837027179,pi/2) q[11];
rzz(2.716104633384914) q[12],q[13];
u3(0.4171245396069413,-1.5696879385819917,pi/2) q[12];
u3(0.4171245396069413,-1.5704040503271197,pi/2) q[13];
u3(0.4171245396069413,-1.5705239110420173,pi/2) q[9];
rzz(-0.6516574039173101) q[0],q[9];
rzz(-0.6516661455609452) q[0],q[10];
rzz(-0.6516587743389354) q[0],q[11];
rzz(-0.6516566685574444) q[0],q[12];
rzz(-0.6516136713833816) q[0],q[13];
rx(11.180437998135138) q[0];
rzz(-0.651616271415974) q[1],q[9];
rzz(-0.6516214328140151) q[1],q[10];
rzz(-0.6516968981303133) q[1],q[11];
rzz(-0.6516515572518444) q[1],q[12];
rzz(-0.651645425347714) q[1],q[13];
rx(11.180437998135138) q[1];
rzz(-0.651654276598488) q[2],q[9];
rzz(-0.6516376894808582) q[2],q[10];
rzz(-0.6516387820236847) q[2],q[11];
rzz(-0.6516487119601868) q[2],q[12];
rzz(-0.6516592731901483) q[2],q[13];
rx(11.180437998135138) q[2];
rzz(-0.6516333279224533) q[3],q[9];
rzz(-0.6516275040357759) q[3],q[10];
rzz(-0.6516328386172217) q[3],q[11];
rzz(-0.6516420084510701) q[3],q[12];
rzz(-0.6516286831896559) q[3],q[13];
rx(11.180437998135138) q[3];
rzz(-0.6516419984281885) q[4],q[9];
rzz(-0.6516387458208547) q[4],q[10];
rzz(-0.6516447900247642) q[4],q[11];
rzz(-0.6516455750826436) q[4],q[12];
rzz(-0.6516563602564717) q[4],q[13];
rx(11.180437998135138) q[4];
rzz(-0.651615536444683) q[5],q[9];
rzz(-0.6516336674399352) q[5],q[10];
rzz(-0.6516344917832826) q[5],q[11];
rzz(-0.6515863782516599) q[5],q[12];
rzz(-0.6516256412222426) q[5],q[13];
rx(11.180437998135138) q[5];
rzz(-0.6516547857637496) q[6],q[9];
rzz(-0.6516503478949076) q[6],q[10];
rzz(-0.65168076208401) q[6],q[11];
rzz(-0.6516329540435486) q[6],q[12];
rzz(-0.651631656688556) q[6],q[13];
rx(11.180437998135138) q[6];
rzz(-0.6516349084168719) q[7],q[9];
rzz(-0.65161704004637) q[7],q[10];
rzz(-0.6516437314406675) q[7],q[11];
rzz(-0.6516362763719541) q[7],q[12];
rzz(-0.6516547136448948) q[7],q[13];
rx(11.180437998135138) q[7];
rzz(-0.6516736526773594) q[8],q[9];
rzz(-0.6516317542947176) q[8],q[10];
rzz(-0.6515964541643968) q[8],q[11];
rzz(-0.6516565312514702) q[8],q[12];
rzz(-0.6516182304253921) q[8],q[13];
rx(11.180437998135138) q[8];
rzz(-0.6516472890504352) q[9],q[10];
rzz(-0.6516411592127351) q[9],q[11];
rzz(-0.6516678619481907) q[10],q[11];
rzz(-0.6516429731046595) q[9],q[12];
rzz(-0.651630554281182) q[10],q[12];
rzz(-0.6516307880676048) q[11],q[12];
rzz(-0.6516469963177334) q[9],q[13];
rzz(-0.6516249943639999) q[10],q[13];
rx(11.180437998135138) q[10];
rzz(-0.651641436471182) q[11],q[13];
rx(11.180437998135138) q[11];
rzz(-0.6516429534935579) q[12],q[13];
rx(11.180437998135138) q[12];
rx(11.180437998135138) q[13];
rx(11.180437998135138) q[9];
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
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13];
meas11[0] = measure q[0];
meas11[1] = measure q[1];
meas11[2] = measure q[2];
meas11[3] = measure q[3];
meas11[4] = measure q[4];
meas11[5] = measure q[5];
meas11[6] = measure q[6];
meas11[7] = measure q[7];
meas11[8] = measure q[8];
meas11[9] = measure q[9];
meas11[10] = measure q[10];
meas11[11] = measure q[11];
meas11[12] = measure q[12];
meas11[13] = measure q[13];