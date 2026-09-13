//! Расчёт стоимости по ценам из секции HTTP-провайдера главного конфига.
//!
//! Все цены задаются в долларах за миллион токенов. `None` означает, что цена
//! модели не настроена и стоимость вызова неизвестна. Если отдельные цены
//! чтения или записи кеша не указаны, для этих токенов применяется цена входа.

use crate::config::ModelPrice;

const TOKENS_PER_MILLION: f64 = 1_000_000.0;

/// Стоимость хода OpenAI-совместимого API. `prompt_tokens` уже включает токены,
/// прочитанные из кеша.
pub fn openai_cost(
    price: Option<&ModelPrice>,
    prompt_tokens: u32,
    cached_tokens: u32,
    completion_tokens: u32,
) -> Option<f64> {
    let price = price?;
    let cached = cached_tokens.min(prompt_tokens);
    let regular = prompt_tokens - cached;
    Some(
        (regular as f64 * price.input
            + cached as f64 * price.cache_read.unwrap_or(price.input)
            + completion_tokens as f64 * price.output)
            / TOKENS_PER_MILLION,
    )
}

/// Стоимость хода Anthropic Messages API. `input_tokens` не включает токены
/// создания и чтения кеша: они приходят отдельными полями.
pub fn anthropic_cost(
    price: Option<&ModelPrice>,
    input_tokens: u32,
    cache_creation: u32,
    cache_read: u32,
    output_tokens: u32,
) -> Option<f64> {
    let price = price?;
    Some(
        (input_tokens as f64 * price.input
            + cache_creation as f64 * price.cache_write.unwrap_or(price.input)
            + cache_read as f64 * price.cache_read.unwrap_or(price.input)
            + output_tokens as f64 * price.output)
            / TOKENS_PER_MILLION,
    )
}

/// Сложить стоимость ходов. Если цена хотя бы одного хода неизвестна,
/// стоимость всего вызова тоже неизвестна.
pub fn add_cost(total: Option<f64>, turn: Option<f64>) -> Option<f64> {
    Some(total? + turn?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn price() -> ModelPrice {
        ModelPrice {
            input: 2.0,
            output: 8.0,
            cache_read: Some(0.5),
            cache_write: Some(3.0),
        }
    }

    #[test]
    fn openai_cost_uses_cache_price() {
        let cost = openai_cost(Some(&price()), 1_000_000, 800_000, 100_000).unwrap();
        assert!((cost - 1.6).abs() < 1e-12, "cost={cost}");
    }

    #[test]
    fn openai_cost_uses_input_price_without_cache_price() {
        let mut price = price();
        price.cache_read = None;
        let cost = openai_cost(Some(&price), 1_000_000, 800_000, 100_000).unwrap();
        assert!((cost - 2.8).abs() < 1e-12, "cost={cost}");
    }

    #[test]
    fn openai_cost_clamps_cached_tokens() {
        let cost = openai_cost(Some(&price()), 1_000, 5_000, 0).unwrap();
        assert!((cost - 0.0005).abs() < 1e-12, "cost={cost}");
    }

    #[test]
    fn openai_cost_without_price_is_unknown() {
        assert_eq!(openai_cost(None, 1, 0, 1), None);
    }

    #[test]
    fn anthropic_cost_uses_cache_prices() {
        let cost = anthropic_cost(Some(&price()), 100_000, 200_000, 300_000, 400_000).unwrap();
        assert!((cost - 4.15).abs() < 1e-12, "cost={cost}");
    }

    #[test]
    fn anthropic_cost_uses_input_price_without_cache_prices() {
        let price = ModelPrice {
            input: 2.0,
            output: 8.0,
            cache_read: None,
            cache_write: None,
        };
        let cost = anthropic_cost(Some(&price), 100_000, 200_000, 300_000, 400_000).unwrap();
        assert!((cost - 4.4).abs() < 1e-12, "cost={cost}");
    }

    #[test]
    fn anthropic_cost_without_price_is_unknown() {
        assert_eq!(anthropic_cost(None, 1, 1, 1, 1), None);
    }

    #[test]
    fn add_cost_propagates_unknown_value() {
        assert_eq!(add_cost(Some(1.0), Some(2.0)), Some(3.0));
        assert_eq!(add_cost(None, Some(2.0)), None);
        assert_eq!(add_cost(Some(1.0), None), None);
        assert_eq!(add_cost(None, None), None);
    }
}
