#ifndef Constellation_h
#define Constellation_h

#include <complex>
#include <cstdint>
#include <map>

namespace cuOFDM {

// 8 bit integer selected as max modulation is QAM256
typedef std::map<uint8_t, std::complex<float>> modMap;

class Constellation {
public:
  Constellation();
  Constellation(const Constellation &aCopy);
  Constellation &operator=(const Constellation &aCopy);
  ~Constellation();

  inline const modMap &getMap() const { return myMap; }

protected:
  modMap myMap;

  virtual void genMap();
};

class QAM256 : public Constellation {
public:
  QAM256();
  QAM256(const QAM256 &aCopy);
  QAM256 &operator=(const QAM256 &aCopy);
  ~QAM256();

protected:
  void genMap();
};

} // namespace cuOFDM

#endif
