string ip_address
string address
int32 frame_sn

string target

# type 1 TLV
string mac

# type 2 TLV
string payload

# type 3 TLV
bool meas_valid
float64 stamp
float64 variance

# type 4 TLV
bool type4_received
float32 rssi
float32 fpp
float32 los
float32 temperature

# type 5 TLV
bool type5_received
int32 rtto
int32 integrator
uint8 channel
uint8 prf
uint8 br
