#include <memory>
#include <vector>
#include <stdint.h>

struct Mat_32
{
  std::shared_ptr<std::vector<uint32_t> > data;
  struct Offset
  {
    int x, y;
    Offset()
      : x(0),
        y(0)
    {};
    Offset(int _x, int _y)
      : x(_x),
        y(_y)
    {};
  };
  Offset offset;
  int rows, cols, step;

  Mat_32(int _rows, int _cols)
    : data(new std::vector<uint32_t>()),
      offset(0, 0),
      rows(_rows),
      cols(_cols),
      step(_cols)
  {};
  Mat_32(int _rows, int _cols, uint32_t _value)
    : data(new std::vector<uint32_t>(static_cast<uint32_t>(_rows * _cols), _value)),
      offset(0, 0),
      rows(_rows),
      cols(_cols),
      step(_cols)
  {};
  Mat_32(Mat_32& _parent, int _x, int _y, int _width, int _height)
    : data(_parent.data),
      offset(_x, _y),
      rows(_height),
      cols(_width),
      step(_parent.step)
  {};

  inline uint32_t& unsafeAt(int _x, int _y)
  {
    return (*data)[(offset.y + _y) * step + _x + offset.x];
  };

  //Determines if the given location (wrt the parent) is within this Mat_32)
  inline bool isWithin(int _parentX, int _parentY)
  {
    return (offset.x <= _parentX && offset.y <= _parentY &&
      (offset.x + cols) > _parentX && (offset.y + rows) > _parentY);
  };
};