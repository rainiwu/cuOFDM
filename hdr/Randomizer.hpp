#ifndef Randomizer_h
#define Randomizer_h

namespace cuOFDM {

/** Randomizer is a channel simulation utility in CUDA
 * Applies random noise to some Modulated channel
 */
class Randomizer {
public:
  Randomizer();
  Randomizer(const Randomizer &aCopy);
  Randomizer &operator=(const Randomizer &aCopy);
  ~Randomizer();

protected:
};
} // namespace cuOFDM

#endif
