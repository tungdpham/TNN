#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace tnn {
template <typename... Types>
class Variant {
public:
  static constexpr size_t data_size = std::max({sizeof(Types)...});
  static constexpr size_t data_alignment = std::max({alignof(Types)...});
  template <typename T>
  static constexpr size_t index_of() {
    size_t index = 0;
    size_t match = 0;
    ((std::is_same_v<T, Types> ? (match = index) : ++index), ...);
    static_assert((std::is_same_v<T, Types> || ...), "Type not found in Variant");
    return match;
  }

  alignas(data_alignment) std::byte storage[data_size];

  Variant()
      : type_index(0) {
    new (storage) typename std::tuple_element<0, std::tuple<Types...>>::type();
  }

  ~Variant() { destroy(); }

  Variant(Variant&& other) noexcept
      : type_index(other.type_index) {
    other.visit([this](auto& value) {
      using T = std::decay_t<decltype(value)>;
      new (storage) T(std::move(value));
    });
  }

  Variant& operator=(Variant&& other) noexcept {
    if (this != &other) {
      destroy();
      type_index = other.type_index;
      other.visit([this](auto& value) {
        using T = std::decay_t<decltype(value)>;
        new (storage) T(std::move(value));
      });
    }
    return *this;
  }

  Variant& operator=(const Variant& other) {
    if (this != &other) {
      destroy();
      type_index = other.type_index;
      other.visit([this](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        new (storage) T(value);
      });
    }
    return *this;
  }

  template <typename T,
            typename = std::enable_if_t<(!std::is_same_v<std::decay_t<T>, Variant> &&
                                         (std::is_same_v<std::decay_t<T>, Types> || ...))>>
  Variant(T&& value) {
    using DecayedT = std::decay_t<T>;
    type_index = get_type_index<DecayedT>();
    new (storage) DecayedT(std::forward<T>(value));
  }

  template <typename T, typename... Args>
  Variant(std::in_place_type_t<T>, Args&&... args) {
    type_index = get_type_index<T>();
    new (storage) T(std::forward<Args>(args)...);
  }

  const uint32_t& index() const { return type_index; }

  template <typename Visitor>
  void visit(Visitor&& visitor) {
    size_t i = 0;
    (..., (type_index == i++ ? visitor(*reinterpret_cast<Types*>(storage)) : void()));
  }

  template <typename Visitor>
  void visit(Visitor&& visitor) const {
    size_t i = 0;
    (..., (type_index == i++ ? visitor(*reinterpret_cast<const Types*>(storage)) : void()));
  }

  template <typename Type>
  bool holds() const {
    return type_index == get_type_index<Type>();
  }

  template <typename Type>
  Type& get() {
    return *reinterpret_cast<Type*>(storage);
  }

  template <typename Type>
  const Type& get() const {
    return *reinterpret_cast<const Type*>(storage);
  }

  template <size_t I = 0>
  void construct_at_index(uint32_t target_index) {
    destroy();
    if constexpr (I < sizeof...(Types)) {
      this->type_index = target_index;
      if (I == target_index) {
        using T = std::tuple_element_t<I, std::tuple<Types...>>;
        new (storage) T();
        return;
      }
      construct_at_index<I + 1>(target_index);
    } else {
      static_assert(I < sizeof...(Types), "Type not found in Variant");
    }
  }

private:
  uint32_t type_index;

  template <typename T, size_t Index = 0>
  static constexpr uint32_t get_type_index() {
    if constexpr (Index >= sizeof...(Types)) {
      static_assert(Index < sizeof...(Types), "Type not found in Variant");
      return 0;  // Should never reach here
    } else if constexpr (std::is_same_v<T, std::tuple_element_t<Index, std::tuple<Types...>>>) {
      return Index;
    } else {
      return get_type_index<T, Index + 1>();
    }
  }

  void destroy() {
    visit([](auto& value) {
      using T = std::decay_t<decltype(value)>;
      value.~T();
    });
  }
};

template <typename Archiver, typename... Types>
void archive(Archiver& archiver, Variant<Types...>& variant) {
  uint32_t old_index = variant.index();
  archiver(variant.index());
  if (old_index != variant.index()) {
    variant.construct_at_index(variant.index());
  }
  variant.visit([&archiver](auto& value) { archiver(value); });
}

template <typename Archiver, typename... Types>
void archive(Archiver& archiver, const Variant<Types...>& variant) {
  archiver(variant.index());
  variant.visit([&archiver](const auto& value) { archiver(value); });
}

}  // namespace tnn