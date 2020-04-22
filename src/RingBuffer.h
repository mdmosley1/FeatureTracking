#include <vector>

template
class<T>
class RingBuffer
{
public:
    RingBuffer(const size_t size);
    size_t capacity() const;
    size_t size() const;
    size_t full() const;
    void load(const T element);
    T read() const;
private:
    size_t head_m, tail_m;
    std::vector<T> buf_m;
};
