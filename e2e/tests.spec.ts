import { test, expect } from '@playwright/test';

test.describe('End-to-End Tests', () => {
  test('T7: Basic Page Load Test', async ({ page }) => {
    await page.goto('http://localhost:5173');
    await expect(page).toHaveTitle(/本机知识库助手/i);
    
    // Verify tabs
    await expect(page.getByRole('button', { name: '对话' })).toBeVisible();
    await expect(page.getByRole('button', { name: '知识库' })).toBeVisible();
    await expect(page.getByRole('button', { name: '设置' })).toBeVisible();
    
    await page.screenshot({ path: 'output/playwright/T7_basic_load.png' });
  });

  test('T8: RAG Q&A Test', async ({ page }) => {
    await page.goto('http://localhost:5173');
    
    // Input question
    const input = page.getByPlaceholder('输入你的问题...');
    await input.fill('什么是测试？');
    await page.keyboard.press('Enter');
    
    // Wait for response
    const answerCard = page.locator('.answer-card');
    await expect(answerCard).toBeVisible({ timeout: 30000 });
    const answerText = answerCard.locator('.markdown-body');
    await expect(answerText).not.toBeEmpty();
    
    await page.screenshot({ path: 'output/playwright/T8_rag_qa.png' });
  });

  test('T9: History Feature Test', async ({ page }) => {
    await page.goto('http://localhost:5173');
    
    // Click history button
    await page.getByRole('button', { name: '📜 历史' }).click();
    
    // Verify sidebar header
    await expect(page.getByRole('heading', { name: '历史会话' })).toBeVisible();
    
    // Wait for loading to finish
    await expect(page.getByText('加载中...')).not.toBeVisible();
    
    await page.screenshot({ path: 'output/playwright/T9_history.png' });
  });

  test('T10: Settings Page Test', async ({ page }) => {
    await page.goto('http://localhost:5173');
    
    // Click settings tab
    await page.getByRole('button', { name: '设置' }).click();
    
    // Verify status
    await expect(page.getByText('后端连接')).toBeVisible();
    await expect(page.getByText('在线', { exact: true })).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('运行模式')).toBeVisible();
    await expect(page.getByText('Mock')).toBeVisible();
    
    await page.screenshot({ path: 'output/playwright/T10_settings.png' });
  });
});
