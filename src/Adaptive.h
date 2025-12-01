// ---------- AdaptUnit.h ----------
#ifndef ADAPT_UNIT_H
#define ADAPT_UNIT_H

#include <cmath>
#include <cstdint>

class AdaptUnit {
private:
    double I_adapt    = 0.0;   // 억제 전류(>=0)
    double tau_adapt  = 100000.0;   // 누수 시정수
    double b_step     = 0.05;   // 스파이크 시 증가량
    uint32_t T_last   = 0;     // 직전 업데이트 시각

public:
    AdaptUnit(double tau_adapt_, double b_step_)
        : tau_adapt(tau_adapt_), b_step(b_step_) {}

    inline void leak(uint32_t T_now) {
        if (T_now == T_last) return;
        // 첫 호출 시 T_last==0일 때 과도 왜곡 방지
        if (T_last == 0) { T_last = T_now; return; }
        I_adapt *= std::exp(-(static_cast<double>(T_now - T_last)) / tau_adapt);
    }
    inline void T_set(uint32_t T_now) {
        T_last   = T_now;
    }

    inline void up() noexcept { I_adapt += b_step; }
    [[nodiscard]] inline double get_I() const noexcept { return I_adapt; }
    inline void reset() noexcept { I_adapt = 0.0; T_last = 0; }
};

#endif // ADAPT_UNIT_H
