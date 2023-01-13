#include <memory>
#include <type_traits>
#include <utility>

namespace numa {
template <typename T, typename A = std::allocator<T>>
class default_init_allocator : public A {
  // Implementation taken from https://stackoverflow.com/a/21028912
  // see also https://hackingcpp.com/cpp/recipe/uninitialized_numeric_array.html
 public:
  using A::A;

  template <typename U>
  struct rebind {
    using other = default_init_allocator<
        U,
        typename std::allocator_traits<A>::template rebind_alloc<U>>;
  };

  template <typename U>
  void construct(U* ptr) noexcept(
      std::is_nothrow_default_constructible<U>::value) {
    ::new (static_cast<void*>(ptr)) U;
  }
  template <typename U, typename... ArgsT>
  void construct(U* ptr, ArgsT&&... args) {
    std::allocator_traits<A>::construct(static_cast<A&>(*this), ptr,
                                        std::forward<ArgsT>(args)...);
  }
};

template <typename T, typename A = std::allocator<T>>
class no_init_allocator : public A {
  // Implementation adapted from https://stackoverflow.com/a/21028912
  // see also https://hackingcpp.com/cpp/recipe/uninitialized_numeric_array.html
 public:
  using A::A;

  template <typename U>
  struct rebind {
    using other = no_init_allocator<
        U,
        typename std::allocator_traits<A>::template rebind_alloc<U>>;
  };

  template <typename U, typename... ArgsT>
  void construct(U* ptr, ArgsT&&... args) { }
};
}  // namespace numa